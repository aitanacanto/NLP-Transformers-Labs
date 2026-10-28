# Setup

Start with Python 3.10 or 3.11. The dependencies are intentionally listed without claiming an untested version lock.

```bash
python -m venv .venv
```

Windows PowerShell: `.venv\Scripts\Activate.ps1`  
macOS/Linux: `source .venv/bin/activate`

```bash
python -m pip install -r requirements.txt
python labs/01-attention/assignment1.py
python labs/02-small-language-model/assignment2.py
python labs/03-toy-transformer/assignment3.py
python labs/04-review-rating-regression/assignment4.py
```

Run one lab at a time. The chatbot waits for terminal input (`exit` ends the conversation). Lab 04 runs two training experiments and uses 100 training examples and 100 test examples. The dataset download is much larger than this selected subset. Downloads and training are not needed to browse the source.
