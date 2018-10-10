import cv2
import numpy as np
vidcap = cv2.VideoCapture('video.mp4')
success,image = vidcap.read()
length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
print('LEN',length )

count = 0
width  = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(width, height)

color_box = {}
out = ''
repeat = 0
total = 0

def process_chunk(w, h):
    chunk = []
    key = ''
    try:
        for i in range(h, h + 8):
            row = []
            for j in range(w, w + 8):
                print(i, j)
                row.append(image[i, j])
                key += str(image[i, j]) + '-'
            chunk.append(row)
    except:
        exit()
    return chunk, key

while success:
    w = 0
    h = 0
    indx = 0
    with open("sample.txt", "w") as text_file:
        np.savetxt(text_file, image)
    if False:
        while h < height and w < width :
            chunk, key = process_chunk(w, h)
            k = str(count) + '@' +str(indx)
            try:
                color_box[key] = (k, chunk)
                out += '-' + key
            except:
                nk, c = color_box[key]
                out += '-' + nk
                repeat +=1
                print("EXISTS KEY")
            indx += 1
            total +=1

            #print(w, h)
            if(w == width and h == height):
                break
            if(w < width):
                w += 8
            else:
                w = 0
                h += 8
        count += 1
        success,image = vidcap.read()

with open("Output.txt", "w") as text_file:
    text_file.write("%s %s %s \n %s" % (str(length), str(repeat), str(total), str(out)))
    print('OUTPUT SAVED')

with open("Dict.txt", "w") as txt_file:
    txt_file.write("%s %s %s \n %s" % (str(length), str(repeat), str(total), str(color_box)))
    print('COLORS SAVED')