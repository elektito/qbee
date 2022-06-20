#
# This script is used to create a PNG character sheet using a
# font. The resulting character sheet is 16x16, containing 256
# characters.
#

from PIL import Image, ImageDraw, ImageFont


font_file = 'Mx437_IBM_VGA_9x16.ttf'
output_file = 'font-9x16.png'
char_width = 9
char_height = 16
bg_color = (0, 0, 0)
fg_color = (255, 255, 255)
grid_rows = 16
grid_cols = 16


def draw_sheet():
    image_size = (char_width * grid_cols, char_height * grid_rows)
    image = Image.new('RGB', image_size, bg_color)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_file, size=char_height)

    for i in range(0, grid_rows):
        for j in range(0, grid_cols):
            x = j * char_width
            y = i * char_height
            char_code = i * grid_cols + j
            char = bytes([char_code]).decode('cp437')
            draw.text((x, y), char, font=font, fill=fg_color)

    image.save(output_file)


if __name__ == '__main__':
    draw_sheet()
