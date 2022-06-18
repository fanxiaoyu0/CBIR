# image = Image.open('../data/image/beach/110.jpg')  
# w, h = image.size  
# colors = image.getcolors(w*h)
# # print(colors)

# def hexencode(rgb):
#     r=rgb[0]
#     g=rgb[1]
#     b=rgb[2]
#     # print('#%02x%02x%02x' % (r,g,b))
#     # fdshjk
#     return '#%02x%02x%02x' % (r,g,b)

# x=[i for i in range(len(colors))]
# y=[i[0] for i in colors]
# color=[hexencode(i[1]) for i in colors]
# # color=
# plt.bar(x,y,color=color)

# # for idx, c in enumerate(colors):
#     # plt.bar(idx, c[0], color=hexencode(c[1]))
# # for idx, c in enumerate(tqdm(colors)):
#     #  plt.bar(idx, c[0], color=hexencode(c[1]),edgecolor=hexencode(c[1]))
# plt.savefig('../result/2.png')
# plt.show()