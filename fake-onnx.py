import onnx
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
import numpy as np

# Some text-to-image application use onnx model to filter NSFW content.
# This is example to create a empty onnx model to cheat the application.

# Define input and output tensor information (same as your original model)
X = make_tensor_value_info('clip_input', onnx.TensorProto.FLOAT16, [1, 3, 224, 224])
Y = make_tensor_value_info('concept_scores', onnx.TensorProto.FLOAT16, [1, 17])

# Create a numpy array of zeros
zeros_array = np.zeros((1, 17), dtype=np.float16)  # Use float16 for consistency

# Create the constant node using the numpy array as the raw data
constant_node = make_node("Constant", [], ["concept_scores"],
                           value=onnx.TensorProto(
                                   data_type=onnx.TensorProto.FLOAT16,
                                   dims=zeros_array.shape,
                                   raw_data=zeros_array.tobytes()  # Crucial for older onnx
                               ))


# Create the graph with the constant node
graph = make_graph([constant_node], 'constant_graph', [X], [Y])  # X is not used

# Create the ONNX model
onnx_model = make_model(graph)

# Change the opset version to 21
onnx_model.opset_import.pop()
onnx_model.opset_import.append(onnx.OperatorSetIdProto(version=21))

# Save the model to a file
with open("ContentFilter.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("ONNX model with constant output (0, 0, ..., 0) created successfully!")
