./game_of_life_cuda
ffmpeg -r 24 -f image2 -r 1/0.4 -i $(IMGS_FOLDER)/%d.png -vcodec mpeg4 -y movie.mp4