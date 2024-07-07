# Dynamic Time Warping (DTW) for Isolated Digit Recognition

This repository contains Python code for implementing Dynamic Time Warping (DTW) to recognize isolated digits from speech signals. DTW is used to compare sequences of MFCC (Mel Frequency Cepstral Coefficients) features extracted from digit recordings.


## Usage

1. **Recording Digits**: Use `record_digit.py` to record digit signals. Ensure the recording is clear and representative of each digit.
2. **Compute MFCC**: Use `compute_mfcc.py` to extract MFCC features from recorded signals.
3. **Train and Test**: Adjust the paths and filenames in `evaluate.py` to train the model using recorded templates and test against new recordings.
4. **Evaluate Accuracy**: Run `evaluate.py` to compute recognition accuracy and generate a confusion matrix.

## Requirements

- Python 3.x
- numpy
- scipy
- matplotlib
- python_speech_features
- speechpy

## Notes

- Ensure recordings are clear and consistent for accurate recognition.
- DTW is effective for comparing sequences of varying lengths and non-linear alignments.

## Authors

- M GANESH

## License

This project is licensed under the MIT License - see the LICENSE file for details.
