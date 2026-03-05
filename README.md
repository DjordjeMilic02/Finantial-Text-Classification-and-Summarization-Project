This repository contains the code and artifacts for the main application:
- input type classification (news / earnings call / central bank speech)
- domain-specific summarization
- domain-specific sentiment classification

The main app entrypoint is:
- `src/main.py` (PySide6 GUI)

## Top-Level Folders

- `src/`
  - Main app:
    - `src/main.py` - Main app used for runing all available models
  - `src/startClassifier/`
    - `src/startClassifier/FINBERTClassifier.py` - Fine-tunes the start/document-type classifier.
    - `src/startClassifier/parseCustomDataset.py` - Parses `customDataset/` text files into JSONL records for classifier training.
    - `src/startClassifier/runClassifier.py` - CLI inference script for the start classifier using a test input file. Used for testing.
    - `src/startClassifier/StartClassifierRunner.py` - Reusable lazy-loaded inference runner used by the main app.
  - `src/summarizers/`
    - `src/summarizers/runnerBART.py` - Runtime BART news summarizer with chunking for long inputs. Used by the main app.
    - `src/summarizers/runnerPEGASUS.py` - Runtime PEGASUS earnings summarizer used in the main app.
    - `src/summarizers/runnerT5.py` - Runtime T5 central-bank summarizer used by the main app.
    - `src/summarizers/summarizerBART.py` - Standalone file-to-file BART summarization script. Used for testing.
    - `src/summarizers/summarizerPEGASUS.py` - Standalone file-to-file PEGASUS summarization script. Used for testing.
    - `src/summarizers/summarizerT5.py` - Standalone file-to-file T5 summarization script. Used for testing.
  - `src/summarizersFineTuning/`
    - `src/summarizersFineTuning/fineTuningBART.py` - Fine-tunes BART for financial news summarization and saves training metrics.
    - `src/summarizersFineTuning/fineTuningPEGASUS.py` - Fine-tunes PEGASUS for earnings call summarization saves training metrics.
    - `src/summarizersFineTuning/fineTuningT5.py` - Fine-tunes T5 for central-bank summarization and exports training metrics.
  - `src/sentimentClassifiers/`
    - `src/sentimentClassifiers/classificationNews.py` - Standalone news sentiment inference script. Used for testing.
    - `src/sentimentClassifiers/classificationCompany.py` - Standalone company/earnings sentiment inference script. Used for testing.
    - `src/sentimentClassifiers/classificationCentralBank.py` - Standalone central-bank sentiment inference script. Used for testing.
    - `src/sentimentClassifiers/classificationCustom.py` - Standalone custom central-bank sentiment inference script. Used for testing.
    - `src/sentimentClassifiers/runnerNews.py` - Reusable lazy-loaded news sentiment runner. Used by the main app.
    - `src/sentimentClassifiers/runnerCompany.py` - Reusable lazy-loaded company/earnings sentiment runner. Used by the main app.
    - `src/sentimentClassifiers/runnerBank.py` - Reusable lazy-loaded central-bank sentiment runner. Used by the main app.
    - `src/sentimentClassifiers/runnerCustom.py` - Reusable lazy-loaded custom HAN sentiment runner. Used by the main app.
  - `src/sentimentClassifiersFineTuning/`
    - `src/sentimentClassifiersFineTuning/fineTuningNews.py` - Fine-tunes a news sentiment classifier and saves training metrics.
    - `src/sentimentClassifiersFineTuning/fineTuningCompany.py` - Fine-tunes an earnings/company sentiment classifier and saves metrics.
    - `src/sentimentClassifiersFineTuning/fineTuningCentralBank.py` - Fine-tunes a central-bank sentiment classifier and saves metrics.
  - `src/customSummarizer/`
    - `src/customSummarizer/buildDataset.py` - Builds and normalizes the summarization dataset used by the model and saves dataset splits.
    - `src/customSummarizer/trainModel.py` - Trains the custom hierarchical RNN summarizer and saves model/run artifacts.
    - `src/customSummarizer/runModel.py` - Standalone inference script for the trained custom hierarchical summarizer. Used for testing.
    - `src/customSummarizer/customRunner.py` - GUI/runtime custom summarizer runner with model-based generation. Used by the main app.
  - `src/customModel/`
    - `src/customModel/buildCustomDataset.py` - Builds combined custom sentiment dataset CSVs including balancing/splitting.
    - `src/customModel/trainCustomModelV2.py` - Trains the custom HAN sentiment model and saves checkpoints/metrics/artifacts.

- `input/`
  - Example input documents for quick manual testing.
  - Intentionally tracked as examples.

- `output/`
  - Example output location used by some scripts during development.
  - Intentionally tracked as examples.

- `customDataset/`
  - Custom small dataset representing currency moves after an interest rate announcement.
  - File format is: {currency}/DDMMYYYY.txt. Every file contains a percentage move in a currency's value and text.

- Model directories used:
  - `finbert-finetuned/` (start classifier)
  - `bart-financial-finetuned-final/` (news summarizer)
  - `pegasus-earnings-fast/` (earnings summarizer)
  - `t5-cb-speeches/` (central bank summarizer)
  - `fullnews-longformer-opendatabay/` (news sentiment)
  - `earnings-aiera-finbert/` (earnings sentiment)
  - `cb-stance-flare/` (central bank sentiment)
  - `scratch_han_artifacts/` (custom sentiment model artifacts)
  - `runs/cb_hier_rnn/cb_hier_rnn_v2/` (custom summarizer artifacts)
  - Every folder contains the original files derived from training/fine_tuning, but actual files and models needed for inference are not 
  - present due to being too large (this includes any .safetensors, .bin, .pt or .model files)

## Top-Level Files

- `requirements-general.txt`
  - Pinned common dependencies.

- `requirements-torch-cpu.txt`
  - CPU-only PyTorch dependencies.

- `requirements-torch-cu121.txt`
  - CUDA 12.1 PyTorch dependencies.

- `requirements.txt`
  - Legacy dependency list kept for backward compatibility.

## Environment Setup

Modern python reccomended (developed on 3.12.7).

CPU setup:

```powershell
pip install -r requirements-general.txt
pip install -r requirements-torch-cpu.txt --index-url https://download.pytorch.org/whl/cpu
```

CUDA 12.1 setup:

```powershell
pip install -r requirements-general.txt
pip install -r requirements-torch-cu121.txt --index-url https://download.pytorch.org/whl/cu121
```

Original app was developed on a NVIDIA RTX 3060ti and an Intel Core I5 11600k processor

## Run

From repository root:

```powershell
.\.venv\Scripts\python.exe src\main.py
```

Or with any Python environment that has dependencies installed:

```powershell
python src/main.py
```
