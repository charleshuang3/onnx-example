import onnx

m = onnx.load("ContentFilter.onnx")

print("\nModel nodes:")
for node in m.graph.node:
    print(node)

print("\nModel input:")
for input in m.graph.input:
    print(input)

print("\nModel output:")
for output in m.graph.output:
    print(output)
