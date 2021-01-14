import pixellib
import tensorflow
import time
from pixellib.semantic import semantic_segmentation

start = time.time()
segment_image = semantic_segmentation()
segment_image.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
segment_image.segmentAsPascalvoc("6-example.jpg", output_image_name="out1.jpg")
end = time.time()
print(f"using time: {end - start:.2f} seconds")
