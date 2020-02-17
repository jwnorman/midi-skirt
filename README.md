# midi-skirt
A tool to overcome bouts of writer's block: randomized rhythms, chord progressions, etc.

## Setup

### Setup virtual environment
The package I use to create the midi, `python-midi` hasn't been updated in 4+ years so it's not Python 3 compatible unfortunately.
```
pyenv install 2.7.8
pyenv virtualenv 2.7.8 midi-skirt
```

### Clone repo
```
git clone https://github.com/jwnorman/midi-skirt.git
cd midi-skirt/
```

### Install requirements
```
pip install -r requirements.txt
```

### Run and tweak examples!
```
iPython -i midi_skirt_examples.py
```
