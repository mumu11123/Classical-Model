import onnxruntime as ort
import numpy as np
import time

sess = ort.InferenceSession("resnet18_cifar10.onnx")

x = np.random.randn(1, 3, 32, 32).astype(np.float32)

# 单次推理
output = sess.run(None, {"input": x})
print("输出形状：", output[0].shape)

# 延迟测试
t0 = time.time()
for _ in range(100):
    sess.run(None, {"input": x})
print("平均延迟(ms)：", (time.time()-t0)/100*1000)
