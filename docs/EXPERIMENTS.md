# Experiment scope

## 01 - Attention

The NumPy implementations demonstrate the underlying matrix operations. Small provided examples compare forward computation and gradients with PyTorch.

## 02 - SmolLM2 conversation

This uses a pretrained model without training it. The existing completion condition searches for an end-of-message marker in the entire decoded context, including prior messages. It can therefore stop before the new reply is complete. This behaviour is preserved and should be reviewed before using the chatbot as a polished demonstration.

## 03 - Toy Transformer

The vocabulary has eight tokens. The task is to retain subject, verb and object tokens while discarding interjections. Validation uses newly generated synthetic examples. It is a deliberately small learning exercise, not a natural-language benchmark. No saved final training log is provided.

## 04 - Review-rating regression

The DistilBERT encoder is frozen; the prediction head is trained to regress rating labels. The labels come from the dataset's 0-4 encoding, rather than explicitly converted 1-5 stars. The small sample is appropriate to describe as an instructional experiment. The two training APIs use different optimisation settings, so their metrics should not be presented as a controlled API-only comparison.

Source syntax has been checked. Pretrained downloads, full training and final metrics have not been reproduced during repository preparation.
