# Download Model 

Before using the code, download the [MODEL](https://www.dropbox.com/scl/fi/e143as8hakhz2emdalqif/best.pt?rlkey=bfbz8rgjqtwxcy68ixht8q36m&st=moez9nu7&dl=0)  and place it in ```weights/```.

# Usage Instructions

To use the code, run the following command:

```bash
python "predict_angle_video.py" --video_path "../Sample_Videos/VNDB_HC_028_01242024_PS_R.mp4" --output_path "output" --hand 'right'
```

## Parameters

- `--video_path`: Path to the input video file.
- `--output_path`: Directory where the output will be saved.
- `--hand`: Specify which hand to analyze (`'right'` or `'left'`).

## Example

This command will process the video located at `../Sample_Videos/VNDB_HC_028_01242024_PS_R.mp4`, analyze the right hand, and save the results in the `output` directory.

If  you are using a Mac computer with MPS support, you shuld modify the code to :

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1

python "predict_angle_video.py" --video_path "../Sample_Videos/VNDB_HC_028_01242024_PS_R.mp4" --output_path "output" --hand 'right'
```