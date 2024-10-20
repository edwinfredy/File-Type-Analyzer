# File Type Analyzer
This project uses a Machine Learning Model (Gradient Boost Machine) to analyze files and predict their filetypes, even in the absence of magic numbers. \
Install dependencies with
```
pip install Flask lightgbm numpy
```
Then run `graphicInference.py` and go to http://localhost:5001/. \
Currently, the project can identify .pdf, .exe, .jpg, .mp3 and .mp4 files.
