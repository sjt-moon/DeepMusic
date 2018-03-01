#!/bin/bash

#
# Prerequisite:
#
#     sudo apt install timidity ffmpeg
#

for filename in "$@"; do
	timidity $filename -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 64k $filename.mp3
done
