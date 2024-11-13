# Python Setup

Follow these steps to set up the project:

## Requirements

- Python 3.11

## Setup

1. **Create a virtual environment** in the root directory(Movemaster\cv-server):

   ```bash
   python3 -m venv env
   ```


2. **Activate the virtual environment**:

   - On **Linux/macOS**:

     ```bash
     source env/bin/activate
     ```

   - On **Windows**:
     ```bash
     .\env\Scripts\activate
     ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the project**
   
   ```bash
   python3 Application.py
   ```
   You can eventually use `--mock` if you want to emulate the camera by a mp4 format file

5. **Deactivate the environment** when done:

   ```bash
   deactivate
   ```

You're all set!
