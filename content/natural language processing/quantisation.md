## what is it?

Quantisation is the process of shrinking a model by rescaling its weights and activations from higher precision numerical formats to lower-precision formats (therefore reducing memory, storage costs and inference latencies)

**Precision Formats:**
- **Full Precision:** Typically FP32 (32-bit), FP16 (16-bit), or BF16 (16-bit variant). These use more bits for the fraction and exponent, allowing for high granularity.
- **Quantised Formats:** Typically Integer formats like **Int8** (8-bit) or **Int4** (4-bit). Recent techniques even push down to **3-bit** or **2-bit** precision.

4-bit quantisation is universally optimal.

## how it works?

To convert a high-precision value ($R$) to a quantised value ($Q$), a **mapping function** is used.
1. **Scaling Factor ($S$):** The ratio between the input range (e.g., FP32 min/max) and the target output range (e.g., Int8 -128 to 127).
2. **Zero Point ($Z$):** A bias value used to ensure that a value of zero in the high-precision range maps exactly to a zero in the quantised range.
3. **The Formula:** $Q = \text{round}(R / S + Z)$.
4. **De-quantisation:** Before final output, the values are mapped back to FP16/FP32 so the software can process them.

**Challenges from this**

LLMs are notoriously difficult to quantise because their weight and activation distributions contain extreme **outliers**.
- **Naive Mapping:** If you include extreme outliers in your scaling range, the central (most common) values get "squeezed" together, losing granularity and causing accuracy loss.
- **Clipping:** You can choose to ignore outliers by setting a smaller range ($\alpha$, $\beta$), but if too many are cut off, accuracy plummets.
- **Calibration:** Finding the "sweet spot" often requires a **calibration data set** and metrics like **KL Divergence** to minimise the difference between the original and quantised distributions.


## when quantisation is applied

### dynamic post-training quantisation (dynamic PTQ)

In Dynamic PTQ, the weights are quantised once after training (e.g., from FP16 to Int8), but the **activations** (the data flowing between layers) remain in high precision until the very moment of computation.

During inference, the system looks at the actual range of values in the current batch of data. It calculates the scaling factor ($S$) and zero-point ($Z$) on-the-fly for that specific input. Since it adapts to every unique input, it handles "outlier" values much better than static methods without needing a calibration dataset.

However, there is a "runtime overhead" because the CPU/GPU must perform extra math to calculate these parameters for every single layer during every forward pass.

This approach works for LSTM or Transformer models where activations can vary wildly.


### static post-training quantisation (static PTQ)

Static PTQ quantises both weights and activations **before** the model is ever deployed. To do this, it needs to know what the "typical" activations look like.

How it's done is that a small set of calibration data (usually 100-500 representative samples) is run though the model. An observer module records the min/max or the distribution of activations at every layer. Based on this data, the scales and zero-points are "frozen". At inference time, the model doesn't need to calculate anything— it'll just use these frozen constants as weights and activations.

Advantage is that it is very fast and efficient for inference as it avoids any extra runtime calculations. The con of this approach is if the calibration data is not representative enough of production/real-world data. This will lead to poor model performance as the ranges of the quantised values are "wrong".


### quantisation-aware training (QAT)

QAT doesn't just "compress" a finished model; it trains the model to function _while being compressed_. It is the "Gold Standard" for accuracy.

During fine-tuning, the weights are kept in high-precision (FP32) to allow for tiny gradient updates, but the **forward pass** simulates the rounding and clipping errors of low precision (Int8/Int4). Since rounding is not differentiable (you can't take a derivative of a step function), QAT uses a **Straight-Through Estimator (STE)**. It essentially "ignores" the rounding during the backward pass, allowing gradients to flow back to the weights as if they were still smooth floating-point values.

The advantage is that the model learns to compensate for quantisation noise, it'll find a set of weights that are robust to being rounded.

However, It is computationally expensive. It is essentially doing a full training or extensive fine-tuning run.


## algorithms

### Bitsandbytes (LLM.int8())
This technique uses **Mixed Precision Decomposition**. It identifies the tiny number of outlier dimensions and keeps them in FP16, while quantising the remaining 99% of weights to Int8. It maintains FP16 accuracy even for massive models but can be slower on smaller models due to implementation overhead.


### GPTQ
It is a "weight-only" quantisation method.

A static PTQ technique that enables extreme quantisation (down to 2-bit). It uses advanced math (based on Optimal Brain Quantisation) to find the best weights. Using **ExLlamaV2 kernels**, it can achieve massive speedups (up to 3.24x) and 5x cost reduction by fitting large models on single GPUs.

### AWQ (Activation-aware Weight Quantisation)

AWQ protects "salient" weights (the top 0.1% to 1% of weights that influence generation the most) by keeping them in high precision. It is more robust across different datasets than GPTQ and performs well on instruction-tuned and multimodal models.

### HQQ (Half-Quadratic Quantisation)

A newer **Dynamic PTQ** method that does _not_ require a calibration dataset. It is incredibly fast, quantising a 70B model in roughly **4 minutes** (compared to hours for GPTQ/AWQ) while maintaining competitive accuracy.




%% Optimised Product Quantisation (need to cover this) %%
