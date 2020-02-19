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

If you need a way to play midi files, one way is to drag and drop into GarageBand or Logic Pro, but there's also a helpful commandline tool for this, `timidity`:
```
brew install timidity
```

with the basic usage: `timidity --adjust-tempo=120 melody.mid`

### Run and tweak examples!
You may want to add the `midi-skirt` directory to your `PYTHONPATH` with `export PYTHONPATH=$PYTHONPATH:<path-to-midi-skirt>`
```
iPython -i midi_skirt_examples.py
```
