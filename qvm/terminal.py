import pyglet
import numpy as np
from functools import lru_cache
from PIL import Image, ImageDraw


# Resource paths are relative to where to python module is
pyglet.resource.path = ['..']


class Terminal(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.char_width = 8
        self.char_height = 16
        self.text_columns = 80
        self.text_lines = 25
        self.text_width = self.text_columns * self.char_width
        self.text_height = self.text_lines * self.char_height
        self.text_buffer = bytearray(
            self.text_lines * self.text_columns * 2)

        font_file = pyglet.resource.file('fonts/font-8x16.png')
        self.font_sheet = Image.open(font_file)
        self.font_sheet_rows = 16
        self.font_sheet_cols = 16

        self.text_img = Image.new(
            'RGB',
            (self.text_width, self.text_height),
            (0, 0, 0),
        )

        self.bg_color = 0
        self.fg_color = 7
        self.color_palette = [
            (0, 0, 0),        # black
            (0, 0, 139),      # blue
            (0, 100, 0),      # green
            (0, 139, 139),    # cyan
            (139, 0, 0),      # red
            (139, 0, 139),    # magenta
            (218, 165, 32),   # orange
            (211, 211, 211),  # white
            (105, 105, 105),  # gray
            (21, 21, 255),    # light blue
            (21, 255, 21),    # light green
            (21, 255, 255),   # light cyan
            (255, 21, 21),    # light red
            (255, 21, 255),   # light magenta
            (255, 255, 0),    # yellow
            (255, 255, 255),  # bright white
        ]

        self.cursor_row = self.cursor_col = 0

        self.fg_color = 14
        self.put_text(b"Hello, World!\r\nfoo")

        self.update_text()

        self.on_draw()

    def clear(self):
        self.text_buffer = bytearray(
            self.text_lines * self.text_columns * 2)

    def put_text(self, text):
        if isinstance(text, str):
            text = text.encode('cp437')
        assert isinstance(text, (bytes, bytearray))
        for char in text:
            idx = self.cursor_row * self.text_columns + self.cursor_col
            idx *= 2

            if idx > len(self.text_buffer) - 2:
                break

            if char == 13:
                self.cursor_col = 0
                continue

            if char == 10:
                self.cursor_row += 1

                if self.cursor_row == self.text_lines:
                    self.scroll()
                    self.cursor_row -= 1

                continue

            fg_color = self.fg_color
            bg_color = self.bg_color
            blink = 0x00
            if fg_color >= 16:
                fg_color -= 16
                blink = 0x80
            attrs = blink | (bg_color << 4) | fg_color
            self.text_buffer[idx] = attrs
            self.text_buffer[idx + 1] = char

            self.cursor_col += 1
            if self.cursor_col == self.text_columns:
                self.cursor_col = 0
                self.cursor_row += 1

                if self.cursor_row == self.text_lines:
                    self.scroll()
                    self.cursor_row -= 1

        self.update_text()

    def scroll(self):
        start = self.text_columns * 2
        empty_line = bytearray(self.text_columns * 2)
        self.text_buffer = self.text_buffer[start:] + empty_line

    def update_text(self):
        draw = ImageDraw.Draw(self.text_img)
        draw.rectangle([0, 0, self.text_width, self.text_height],
                       fill=self.bg_color)

        for i in range(0, len(self.text_buffer), 2):
            line = (i // 2) // self.text_columns
            column = (i // 2) % self.text_columns
            x = column * self.char_width
            y = line * self.char_height

            attrs = self.text_buffer[i]
            fg_color = attrs & 0x0f
            bg_color = (attrs & 0x70) >> 4
            blink = (attrs & 0x80) >> 7
            fg_color = self.color_palette[fg_color]
            bg_color = self.color_palette[bg_color]
            char_code = self.text_buffer[i + 1]
            char_img = self._get_char(
                char_code, fg_color, bg_color)

            self.text_img.paste(char_img, (x, y))

        raw_image = self.text_img.tobytes()
        self.text_img_data = pyglet.image.ImageData(
            self.text_width, self.text_height,
            'RGB',
            raw_image,
            pitch=-self.text_width * 3)

        self.text_texture = pyglet.image.Texture.create(
            self.text_width, self.text_height,
            mag_filter = pyglet.image.GL_NEAREST,
        )
        self.text_texture.blit_into(self.text_img_data, 0, 0, 0)

    @lru_cache(maxsize=256)
    def _get_char(self, char_code, fg_color, bg_color):
        assert len(fg_color) == len(bg_color) == 3

        sheet_row = char_code // self.font_sheet_cols
        sheet_col = char_code % self.font_sheet_cols
        cx = sheet_col * self.char_width
        cy = sheet_row * self.char_height
        char_img = self.font_sheet.crop(
            (cx, cy, cx + self.char_width, cy + self.char_height))

        # convert whites to foreground color and blacks to background
        # color
        data = np.array(char_img)
        r, g, b = data.T
        white_areas = (r == 255) & (g == 255) & (b == 255)
        black_areas = (r == 0) & (g == 0) & (b == 0)
        data[...][white_areas.T] = fg_color
        data[...][black_areas.T] = bg_color
        char_img = Image.fromarray(data)

        return char_img

    def on_draw(self):
        self.clear()
        self.text_img_data.blit(0, 0)
        self.text_texture.blit(
            0, 0, width=self.width, height=self.height)

    def on_key_press(self, symbol, modifiers):
        if 32 <= symbol <= 127:
            if (modifiers & pyglet.window.key.LSHIFT or \
                modifiers & pyglet.window.key.RSHIFT) and \
                ord('a') <= symbol <= ord('z'):
                symbol -= 32
            self.text_buffer[100 * 2 + 0] = 0x07
            self.text_buffer[100 * 2 + 1] = symbol
            self.update_text()

        self.put_text(f'foo {symbol}\r\n'.encode('cp437'))


def main():
    window = TerminalWindow(
        caption='QVM Terminal',
        resizable=True,
    )

    pyglet.app.run()


if __name__ == '__main__':
    main()
