
import keras2onnx
import onnx
import keras

keras_app_models = [
    {
        'name': 'vgg16',
        'model_fun': keras.applications.vgg16.VGG16,
        'shape': (224, 224, 3),
        'classes': 1000,
    },
    {
        'name': 'vgg19',
        'model_fun': keras.applications.VGG19,
        'shape': (224, 224, 3),
        'classes': 1000,
    },
    {
        'name': 'resnet50',
        'model_fun': keras.applications.ResNet50,
        'shape': (224, 224, 3),
        'classes': 1000,
    },
    {
        'name': 'resnet101',
        'model_fun': keras.applications.ResNet101,
        'shape': (224, 224, 3),
        'classes': 1000,
    },
    {
        'name': 'resnet152',
        'model_fun': keras.applications.ResNet152,
        'shape': (224, 224, 3),
        'classes': 1000,
    },
    {
        'name': 'resnet50v2',
        'model_fun': keras.applications.ResNet50V2,
        'shape': (299, 299, 3),
        'classes': 1000,
    },
    {
        'name': 'resnet101v2',
        'model_fun': keras.applications.ResNet101V2,
        'shape': (299, 299, 3),
        'classes': 1000,
    },
    {
        'name': 'resnet152v2',
        'model_fun': keras.applications.ResNet152V2,
        'shape': (299, 299, 3),
        'classes': 1000,
    },
    {
        'name': 'inceptionv3',
        'model_fun': keras.applications.InceptionV3,
        'shape': (299, 299, 3),
        'classes': 1000,
    },
    
    {
        'name': 'inception_resnetv2',
        'model_fun': keras.applications.InceptionResNetV2,
        'shape': (299, 299, 3),
        'classes': 1000,
    },
    {
        'name': 'xception',
        'model_fun': keras.applications.Xception,
        'shape': (299, 299, 3),
        'classes': 1000,
    },
    {
        'name': 'mobilenet_alpha_0.25',
        'model_fun': keras.applications.MobileNet,
        'shape': (224, 224, 3),
        'alpha': 0.25,
        'classes': 1000
    },
    {
        'name': 'mobilenet_alpha_0.50',
        'model_fun': keras.applications.MobileNet,
        'shape': (224, 224, 3),
        'alpha': 0.50,
        'classes': 1000
    },
    {
        'name': 'mobilenet_alpha_0.75',
        'model_fun': keras.applications.MobileNet,
        'shape': (224, 224, 3),
        'alpha': 0.75,
        'classes': 1000
    },
    {
        'name': 'mobilenet_alpha_1.0',
        'model_fun': keras.applications.MobileNet,
        'shape': (224, 224, 3),
        'alpha': 1.0,
        'classes': 1000
    },
    {
        'name': 'mobilenetv2_alpha_0.35',
        'model_fun': keras.applications.MobileNetV2,
        'shape': (224, 224, 3),
        'alpha': 0.35,
        'classes': 1000
    },
    {
        'name': 'mobilenetv2_alpha_0.50',
        'model_fun': keras.applications.MobileNetV2,
        'shape': (224, 224, 3),
        'alpha': 0.50,
        'classes': 1000
    },
    {
        'name': 'mobilenetv2_alpha_0.75',
        'model_fun': keras.applications.MobileNetV2,
        'shape': (224, 224, 3),
        'alpha': 0.75,
        'classes': 1000
    },
    {
        'name': 'mobilenetv2_alpha_1.0',
        'model_fun': keras.applications.MobileNetV2,
        'shape': (224, 224, 3),
        'alpha': 1.0,
        'classes': 1000
    },
    {
        'name': 'mobilenetv2_alpha_1.3',
        'model_fun': keras.applications.MobileNetV2,
        'shape': (224, 224, 3),
        'alpha': 1.3,
        'classes': 1000
    },
    {
        'name': 'mobilenetv2_alpha_1.4',
        'model_fun': keras.applications.MobileNetV2,
        'shape': (224, 224, 3),
        'alpha': 1.4,
        'classes': 1000
    },
    {
        'name': 'densenet121',
        'model_fun': keras.applications.DenseNet121,
        'shape': (224, 224, 3),
        'classes': 1000
    },
    {
        'name': 'densenet169',
        'model_fun': keras.applications.DenseNet169,
        'shape': (224, 224, 3),
        'classes': 1000
    },
    {
        'name': 'densenet201',
        'model_fun': keras.applications.DenseNet201,
        'shape': (224, 224, 3),
        'classes': 1000
    },
    {
        'name': 'nasnet_large',
        'model_fun': keras.applications.NASNetMobile,
        'shape': (331, 331, 3),
        'classes': 1000,
    },
    {
        'name': 'nasnet_mobile',
        'model_fun': keras.applications.NASNetMobile,
        'shape': (224, 224, 3),
        'classes': 1000,
    }
]


from onnx import helper




for m in keras_app_models:
    inputs = keras.layers.Input(shape=m["shape"])
    model = m['model_fun'](weights='imagenet', input_tensor=inputs, input_shape=m["shape"], classes=m["classes"], backend=keras.backend)
    onnx_model = keras2onnx.convert_keras(model, m['name'])
    
    print(helper.printable_graph(onnx_model.graph))
    break
    # onnx.save(onnx_model, "imagenet_" + m['name'] + ".onnx")