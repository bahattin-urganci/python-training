from youtube_dl import YoutubeDL
ydl_opts = {
    'verbose': True,
    'format': 'bestaudio/best',  # choice of quality
    #'outtmpl': 'example.%(ext)s',  # name the location
    'noplaylist': True,        # only download single song, not playlist

    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
    }],


}
ydl = YoutubeDL(ydl_opts)

ydl.download(['https://www.youtube.com/watch?v=KKesURNEpHg','https://www.youtube.com/watch?v=zVz9GcqBUoQ'])
