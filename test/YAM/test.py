import cv2

vidcap = cv2.VideoCapture('video.mp4')
success,image = vidcap.read()
length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
print('LEN',length )
count = 0
width  = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
color_box = {}
out = ''
repeat = 0
total = 0

def process_frame(image, indx, w, h):
    global out, repeat, total
    key = str(count) + '@' +str(indx)
    chunk = []
    total += 1
    has = ''
    try:
        for i in range(h, h + 8):
            row = []
            for j in range(w, w + 8):
                row.append(image[i, j])
                has += str(image[i, j]) + '-'
            chunk.append(row)
        try:
            color_box[has] = (key, chunk)
            out += '-' + key
        except:
            k, c = color_box[has]
            print("EXISTS")
            out += '-' + k
            repeat += 1
            
        if w < width and h < height:
            new_indx = indx + 1
            process_frame(image, new_indx, w+8 ,h+8)
    except:
        out += '$'
        print('FRAME PROCESSED', count)

while success:
    w = 0
    h = 0
    indx = 0
    process_frame(image, indx, w, h)
    count += 1
    success,image = vidcap.read()

with open("Output.txt", "w") as text_file:
    text_file.write("%s %s %s \n %s" % (str(length), str(repeat), str(total), out))
    print('OUTPUT SAVED')

with open("Dict.txt", "w") as txt_file:
    txt_file.write("%s %s %s \n %s" % (str(length), str(repeat), str(total), str(color_box)))
    print('COLORS SAVED')