

import importlib
from importlib import reload, import_module
import json
import os
import sys
import arcpy
sys.path.append(os.path.dirname(__file__))
import numpy as np


def get_available_device(max_memory=0.8):
    '''
    select available device based on the memory utilization status of the device
    :param max_memory: the maximum memory utilization ratio that is considered available
    :return: GPU id that is available, -1 means no GPU is available/uses CPU, if GPUtil package is not installed, will
    return 0 
    '''
    try:
        import GPUtil
    except ModuleNotFoundError:
        return 0

    GPUs = GPUtil.getGPUs()
    freeMemory = 0
    available = 0
    for GPU in GPUs:
        if GPU.memoryUtil > max_memory:
            continue
        if GPU.memoryFree >= freeMemory:
            freeMemory = GPU.memoryFree
            available = GPU.id

    return available

features = {
    'displayFieldName': '',
    'fieldAliases': {
        'FID': 'FID',
        'Class': 'Class',
        'Confidence': 'Confidence'
    },
    'geometryType': 'esriGeometryPolygon',
    'fields': [
        {
            'name': 'FID',
            'type': 'esriFieldTypeOID',
            'alias': 'FID'
        },
        {
            'name': 'Class',
            'type': 'esriFieldTypeString',
            'alias': 'Class'
        },
        {
            'name': 'Confidence',
            'type': 'esriFieldTypeDouble',
            'alias': 'Confidence'
        }
    ],
    'features': []
}

fields = {
    'fields': [
        {
            'name': 'OID',
            'type': 'esriFieldTypeOID',
            'alias': 'OID'
        },
        {
            'name': 'Class',
            'type': 'esriFieldTypeString',
            'alias': 'Class'
        },
        {
            'name': 'Confidence',
            'type': 'esriFieldTypeDouble',
            'alias': 'Confidence'
        },
        {
            'name': 'Shape',
            'type': 'esriFieldTypeGeometry',
            'alias': 'Shape'
        }
    ]
}

class GeometryType:
    Point = 1
    Multipoint = 2
    Polyline = 3
    Polygon = 4


class ArcGISObjectClassifier:
    def __init__(self):
        self.name = 'Object classifier'
        self.description = 'This python raster function applies deep learning model to classify objects from overlaid imagery'

    def initialize(self, **kwargs):

        if 'model' not in kwargs:
            return

        # Read esri model definition (emd) file
        model = kwargs['model']
        model_as_file = True

        try:
            with open(model, 'r') as f:
                self.json_info = json.load(f)
        except FileNotFoundError:
            try:
                self.json_info = json.loads(model)
                model_as_file = False
            except json.decoder.JSONDecodeError:
                raise Exception("Invalid model argument")

        sys.path.append(os.path.dirname(__file__))

        framework = self.json_info['Framework']
        if 'ModelConfiguration' in self.json_info:
            modelconfig = self.json_info['ModelConfiguration']
            if isinstance(modelconfig, str):
                if modelconfig not in sys.modules:
                    ChildModelDetector = getattr(import_module(
                        '{}.{}'.format(framework, modelconfig)), 'ChildObjectDetector')
                else:
                    ChildModelDetector = getattr(reload(
                        '{}.{}'.format(framework, modelconfig)), 'ChildObjectDetector')
            else:
                modelconfig = self.json_info['ModelConfiguration']['Name']
                ChildModelDetector = getattr(importlib.import_module(
                    '{}.{}'.format(framework, modelconfig)), 'ChildObjectDetector')
        else:
            raise Exception("Invalid model configuration")

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        device = None
        if 'device' in kwargs:
            device = kwargs['device']
            if device == -2:
                device = get_available_device()

        if device is not None:
            if device >= 0:
                try:
                    import torch
                except Exception:
                    raise Exception("PyTorch is not installed. Install it using conda install -c esri deep-learning-essentials")
                torch.cuda.set_device(device)
                arcpy.env.processorType = "GPU"
                arcpy.env.gpuId = str(device)
            else:
                arcpy.env.processorType = "CPU"

        self.child_object_detector = ChildModelDetector()
        self.child_object_detector.initialize(model, model_as_file)


    def getParameterInfo(self):

        # PRF needs values of these parameters from gp tool user,
        # either from gp tool UI or emd (a json) file.
        required_parameters = [
            {
                # To support mini batch, it is required that Classify Objects Using Deep Learning geoprocessing Tool
                # passes down a stack of raster tiles to PRF for model inference, the keyword required here is 'rasters'.
                'name': 'rasters',
                'dataType': 'rasters',
                'value': None,
                'required': True,
                'displayName': "Rasters",
                'description': 'The collection of overlapping rasters to objects to be classified'
            },
            {
                'name': 'model',
                'dataType': 'string',
                'required': True,
                'displayName': 'Input Model Definition (EMD) File',
                'description': 'Input model definition (EMD) JSON file'
            },
            {
                'name': 'device',
                'dataType': 'numeric',
                'required': False,
                'displayName': 'Device ID',
                'description': 'Device ID'
            }
        ]

        if 'BatchSize' not in self.json_info:
             required_parameters.append(
                 {
                     'name': 'batch_size',
                     'dataType': 'numeric',
                     'required': False,
                     'value': 4,
                     'displayName': 'Batch Size',
                     'description': 'Batch Size'
                 }
             )

        return self.child_object_detector.getParameterInfo(required_parameters)


    def getConfiguration(self, **scalars):


        # The information PRF returns to the GP tool,
        # the information is either from emd or defined in getConfiguration method.

        configuration = self.child_object_detector.getConfiguration(**scalars)

        if 'DataRange' in self.json_info:
            configuration['dataRange'] = tuple(self.json_info['DataRange'])

        configuration['inheritProperties'] = 2|4|8
        configuration['inputMask'] = True


        return configuration

    def getFields(self):

        fields = {
                'fields': [
                    {
                        'name': 'OID',
                        'type': 'esriFieldTypeOID',
                        'alias': 'OID'
                    },
                    {
                        'name': 'Class',
                        'type': 'esriFieldTypeString',
                        'alias': 'Class'
                    },
                    {
                        'name': 'Confidence',
                        'type': 'esriFieldTypeDouble',
                        'alias': 'Confidence'
                    },
                    {
                        'name': 'Shape',
                        'type': 'esriFieldTypeGeometry',
                        'alias': 'Shape'
                    }
                ]
            }
        fields['fields'].append(
            {
                'name': 'Label',
                'type': 'esriFieldTypeString',
                'alias': 'Label'
            }
        )

        if "MetaDataMode" in self.json_info and self.json_info["MetaDataMode"] == "MultiLabeled_Tiles":
            for item in fields['fields']:
                if item['name'] == 'Confidence':
                    item['type'] = 'esriFieldTypeString'

        return json.dumps(fields)

    def getGeometryType(self):
        return GeometryType.Polygon

    def vectorize(self, **pixelBlocks):

        # set pixel values in invalid areas to 0
        rasters_mask = pixelBlocks['rasters_mask']
        rasters_pixels = pixelBlocks['rasters_pixels']

        for i in range(0, len(rasters_pixels)):
            rasters_pixels[i][np.where(rasters_mask[i] == 0)] = 0

        pixelBlocks['rasters_pixels'] = rasters_pixels

        polygon_list, scores, labels = self.child_object_detector.vectorize(**pixelBlocks)

        features['features'] = []

        features['fieldAliases'].update({
            'Label':'Label'
        })

        features['fields'].append(
            {
                'name': 'Label',
                'type': 'esriFieldTypeString',
                'alias': 'Label'
            }
        )

        if "MetaDataMode" in self.json_info and self.json_info["MetaDataMode"] == "MultiLabeled_Tiles":
            for item in features['fields']:
                if item['name'] == 'Confidence':
                    item['type'] = 'esriFieldTypeString'

        for i in range(len(polygon_list)):

            rings = [[]]
            for j in range(len(polygon_list[i])):
                rings[0].append(
                    [
                        polygon_list[i][j][1],
                        polygon_list[i][j][0]
                    ]
                )

            features['features'].append({
                'attributes': {
                    'OID': i + 1,
                    'Confidence': str(scores[i]),
                    'Label': labels[i],
                    'Classname': labels[i]
                },
                'geometry': {
                    'rings': rings
                }
            })

        return {'output_vectors': json.dumps(features)}
