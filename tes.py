
import keras2onnx
import onnx
import keras
from onnx import helper


keras_app_models = [
    # {
    #     'name': 'vgg16',
    #     'model_fun': keras.applications.vgg16.VGG16,
    #     'classes': 1000,
    # },
    # {
    #     'name': 'vgg19',
    #     'model_fun': keras.applications.VGG19,
    #     'classes': 1000,
    # },
    # {
    #     'name': 'resnet50',
    #     'model_fun': keras.applications.ResNet50,
    #     'classes': 1000,
    # },
    # {
    #     'name': 'resnet101',
    #     'model_fun': keras.applications.ResNet101,
    #     'classes': 1000,
    # },
    # {
    #     'name': 'resnet152',
    #     'model_fun': keras.applications.ResNet152,
    #     'classes': 1000,
    # },
    # {
    #     'name': 'resnet50v2',
    #     'model_fun': keras.applications.ResNet50V2,
    #     'classes': 1000,
    # },
    # {
    #     'name': 'resnet101v2',
    #     'model_fun': keras.applications.ResNet101V2,
    #     'classes': 1000,
    # },
    # {
    #     'name': 'resnet152v2',
    #     'model_fun': keras.applications.ResNet152V2,
    #     'shape': (244, 244, 3),
    #     'classes': 1000,
    # },
    # {
    #     'name': 'inceptionv3',
    #     'model_fun': keras.applications.InceptionV3,
    #     'classes': 1000,
    # },
    
    # {
    #     'name': 'inception_resnetv2',
    #     'model_fun': keras.applications.InceptionResNetV2,
    #     'classes': 1000,
    # },
    # {
    #     'name': 'xception',
    #     'model_fun': keras.applications.Xception,
    #     'classes': 1000,
    # },
    {
        'name': 'mobilenet_alpha_0.25',
        'model_fun': keras.applications.MobileNet,
        'alpha': 0.25,
        'classes': 1000
    },
    {
        'name': 'mobilenet_alpha_0.50',
        'model_fun': keras.applications.MobileNet,
        'alpha': 0.50,
        'classes': 1000
    },
    {
        'name': 'mobilenet_alpha_0.75',
        'model_fun': keras.applications.MobileNet,
        'alpha': 0.75,
        'classes': 1000
    },
    {
        'name': 'mobilenet_alpha_1.0',
        'model_fun': keras.applications.MobileNet,
        'alpha': 1.0,
        'classes': 1000
    },
    {
        'name': 'mobilenetv2_alpha_0.35',
        'model_fun': keras.applications.MobileNetV2,
        'alpha': 0.35,
        'classes': 1000
    },
    {
        'name': 'mobilenetv2_alpha_0.50',
        'model_fun': keras.applications.MobileNetV2,
        'alpha': 0.50,
        'classes': 1000
    },
    {
        'name': 'mobilenetv2_alpha_0.75',
        'model_fun': keras.applications.MobileNetV2,
        'alpha': 0.75,
        'classes': 1000
    },
    {
        'name': 'mobilenetv2_alpha_1.0',
        'model_fun': keras.applications.MobileNetV2,
        'alpha': 1.0,
        'classes': 1000
    },
    {
        'name': 'mobilenetv2_alpha_1.3',
        'model_fun': keras.applications.MobileNetV2,
        'alpha': 1.3,
        'classes': 1000
    },
    {
        'name': 'mobilenetv2_alpha_1.4',
        'model_fun': keras.applications.MobileNetV2,
        'alpha': 1.4,
        'classes': 1000
    },
    {
        'name': 'densenet121',
        'model_fun': keras.applications.DenseNet121,
        'classes': 1000
    },
    {
        'name': 'densenet169',
        'model_fun': keras.applications.DenseNet169,
        'classes': 1000
    },
    {
        'name': 'densenet201',
        'model_fun': keras.applications.DenseNet201,
        'classes': 1000
    },
    {
        'name': 'nasnet_large',
        'model_fun': keras.applications.NASNetMobile,
        'classes': 1000,
    },
    {
        'name': 'nasnet_mobile',
        'model_fun': keras.applications.NASNetMobile,
        'classes': 1000,
    }
]

for m in keras_app_models:
    shape = (224, 224, 3)
    weights = None
    # weights = 'imagenet'
    if 'shape' in m:
        shape = m['shape']

    inputs = keras.layers.Input(shape=shape)
    if 'alpha' in m:
        model = m['model_fun'](weights=weights, input_tensor=inputs, input_shape=shape, classes=m["classes"], backend=keras.backend, alpha=m['alpha'])
    else:
        model = m['model_fun'](weights=weights, input_tensor=inputs, input_shape=shape, classes=m["classes"], backend=keras.backend)
    onnx_model = keras2onnx.convert_keras(model, m['name'])
    graph = helper.printable_graph(onnx_model.graph)
    with open("imagenet" + "_" + m["name"] + ".graph", 'w') as _file:
        _file.write(graph)
    onnx.save(onnx_model, "imagenet" + "_" + m["name"] + ".onnx")