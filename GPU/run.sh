./game_of_life_cuda && ffmpeg -r 24 -f image2 -r 1/0.4 -i imgs/%d.png -vcodec mpeg4 -y game_of_life_cuda.mp4 && echo "Success! Video 'game_of_life_cuda.mp4' was generated"
