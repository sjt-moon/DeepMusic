# DeepMusic
RNN based music generator, leveraging GPU power by minibatch of size 128 per iteration.

## Ready-to-use executable scripts

- `datapp.py`: data preprocessing, make pickled datasources from "input.txt"
- `plot-losses-from-pickles.py`: generate 8x6 loss figures from result pickle files
- `generate-music-from-pickles.py`: generate music (`abc` format) from result pickle files
- `midi2mp3.sh`: transform `midi` file to `mp3` file; usage: `./midi2mp3.sh *.m`; prerequisite: `abcmidi`, `timidity`

For detailed usage please check their `-h/--help` options.

Furthermore, `abcmidi` package can be used to transform between `abc` format and `midi` format. Installation:

	sudo apt install abcmidi timidity

Example usage:

	for filename in *.abc; do abc2midi $filename; done

