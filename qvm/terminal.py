import pyglet
import numpy as np
from queue import Empty
from functools import lru_cache
from PIL import Image, ImageDraw


KEYBOARD_BUF_SIZE = 8


# Resource paths are relative to where to python module is
pyglet.resource.path = ['..']
pyglet.resource.reindex()


class TerminalWindow(pyglet.window.Window):
    def __init__(self, request_queue, result_queue, *args, **kwargs):
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

        self.show_cursor = False

        # cursor top and bottom rows
        self.cursor_start = self.char_height - 2
        self.cursor_end = self.char_height

        # the amount of time (in seconds) out of one second the cursor
        # is shown before its hidden for blinking
        self.cursor_blink_show_time = 0.8

        # cursor position on the screen
        self.cursor_row = self.cursor_col = 0

        self.update_text()
        self._text_updated = False

        self.keyboard_buf = []

        self.request_queue = request_queue
        self.result_queue = result_queue
        pyglet.clock.schedule_interval(self.update, 1 / 60)

        self.time = 0.0

    def set(self, attr_name, value):
        setattr(self, attr_name, value)

    def get(self, attr_name):
        return getattr(self, attr_name)

    def update(self, dt):
        self.time += dt
        try:
            while req := self.request_queue.get_nowait():
                method_name, args, kwargs, with_result = req
                result = getattr(self, method_name)(*args, **kwargs)
                if with_result:
                    self.result_queue.put(result)
        except Empty:
            pass

    def clear_screen(self):
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

        self._text_updated = True

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
        text_img_data = pyglet.image.ImageData(
            self.text_width, self.text_height,
            'RGB',
            raw_image,
            pitch=-self.text_width * 3)

        self.text_texture = pyglet.image.Texture.create(
            self.text_width, self.text_height,
            mag_filter = pyglet.image.GL_NEAREST,
        )
        self.text_texture.blit_into(text_img_data, 0, 0, 0)

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
        if self._text_updated:
            self.update_text()
            self._text_updated = False
        self.clear()
        self.text_texture.blit(
            0, 0, width=self.width, height=self.height)

        if 0 <= (self.time % 1.0) <= self.cursor_blink_show_time:
            text_bottom = self.text_texture.height - 1
            x = self.cursor_col * self.char_width
            y = text_bottom - (self.cursor_row * self.char_height + self.cursor_start)
            w = self.char_width
            h = self.cursor_end - self.cursor_start

            h *= self.height / self.text_texture.height
            y *= self.height / self.text_texture.height
            w *= self.width / self.text_texture.width
            x *= self.width / self.text_texture.width

            color = self.color_palette[self.fg_color]
            rect = pyglet.shapes.Rectangle(x, y, w, h, color=color)
            rect.draw()

    def on_key_press(self, symbol, modifiers):
        key = pyglet.window.key

        # don't take into account caps lock and num lock
        modifiers &= ~key.MOD_CAPSLOCK
        modifiers &= ~key.MOD_NUMLOCK

        # convert numpad keys to normal keys
        numpad_keys = {
            key.NUM_HOME: key.HOME,
            key.NUM_END: key.END,
            key.NUM_INSERT: key.INSERT,
            key.NUM_DELETE: key.DELETE,
            key.NUM_DIVIDE: key.SLASH,
            key.NUM_MULTIPLY: key.ASTERISK,
            key.NUM_SUBTRACT: key.MINUS,
            key.NUM_ENTER: key.ENTER,
            key.NUM_DECIMAL: key.PERIOD,
            key.NUM_ADD: key.PLUS,
            key.NUM_UP: key.UP,
            key.NUM_DOWN: key.DOWN,
            key.NUM_LEFT: key.LEFT,
            key.NUM_RIGHT: key.RIGHT,
        }
        if symbol in numpad_keys:
            symbol = numpad_keys[symbol]

        if (ord('a') <= symbol <= ord('z')) and \
           modifiers & key.MOD_SHIFT:
            self._add_key_to_buffer(symbol - (ord('a') - ord('A')))
        elif symbol == key.BACKSPACE:
            if modifiers & key.MOD_ALT:
                self._add_key_to_buffer(127)
            elif modifiers == 0:
                self._add_key_to_buffer(8)
        elif symbol == 0x1700000000:
            # SHIFT+TAB; at least on my machine!
            self._add_key_to_buffer((0, 15))
        elif symbol == key.DELETE and modifiers in (0, key.MOD_SHIFT):
            self._add_key_to_buffer((0, 83))
        elif symbol == key.END and modifiers in (0, key.MOD_SHIFT):
            self._add_key_to_buffer((0, 79))
        elif symbol == key.END and modifiers == key.MOD_CTRL:
            self._add_key_to_buffer((0, 117))
        elif symbol == key.ENTER and modifiers == 0:
            self._add_key_to_buffer(13)
        elif symbol == key.ENTER and modifiers & key.MOD_CTRL:
            self._add_key_to_buffer(10)
        elif symbol in [key.F1, key.F2, key.F3, key.F4, key.F5, key.F6,
                        key.F7, key.F8, key.F9, key.F10]:
            idx = [key.F1, key.F2, key.F3, key.F4, key.F5,
                   key.F6, key.F7, key.F8, key.F9, key.F10].index(symbol)
            k = 59 + idx
            if modifiers & key.MOD_SHIFT:
                k += 84 - 59
            elif modifiers & key.MOD_CTRL:
                k += 94 - 59
            elif modifiers & key.MOD_ALT:
                k += 104 - 59
            self._add_key_to_buffer((0, k))
        elif symbol == key.F11:
            if modifiers == 0:
                self._add_key_to_buffer((0, 133))
            elif modifiers == key.MOD_SHIFT:
                self._add_key_to_buffer((0, 135))
            elif modifiers == key.MOD_CTRL:
                self._add_key_to_buffer((0, 137))
            elif modifiers == key.MOD_ALT:
                self._add_key_to_buffer((0, 139))
        elif symbol == key.F12:
            if modifiers == 0:
                self._add_key_to_buffer((0, 134))
            elif modifiers == key.MOD_SHIFT:
                self._add_key_to_buffer((0, 136))
            elif modifiers == key.MOD_CTRL:
                self._add_key_to_buffer((0, 138))
            elif modifiers == key.MOD_ALT:
                self._add_key_to_buffer((0, 140))
        elif symbol == key.HOME and modifiers in (0, key.MOD_SHIFT):
            self._add_key_to_buffer((0, 71))
        elif symbol == key.HOME and modifiers == key.MOD_CTRL:
            self._add_key_to_buffer((0, 119))
        elif symbol == key.INSERT and modifiers in (0, key.MOD_SHIFT):
            self._add_key_to_buffer((0, 82))
        elif symbol == key.PAGEDOWN and modifiers in (0, key.MOD_SHIFT):
            self._add_key_to_buffer((0, 81))
        elif symbol == key.PAGEDOWN and modifiers == key.MOD_CTRL:
            self._add_key_to_buffer((0, 118))
        elif symbol == key.PAGEUP and modifiers in (0, key.MOD_SHIFT):
            self._add_key_to_buffer((0, 73))
        elif symbol == key.PAGEUP and modifiers == key.MOD_CTRL:
            self._add_key_to_buffer((0, 132))
        elif symbol == key.TAB and modifiers == 0:
            self._add_key_to_buffer(9)
        elif symbol == key.ESCAPE:
            self._add_key_to_buffer(27)
        elif symbol == key.UP:
            if modifiers == 0:
                self._add_key_to_buffer((0, 72))
        elif symbol == key.LEFT:
            if modifiers & key.MOD_CTRL:
                self._add_key_to_buffer((0, 115))
            elif modifiers == 0:
                self._add_key_to_buffer((0, 75))
        elif symbol == key.DOWN:
            if modifiers & key.MOD_CTRL:
                self._add_key_to_buffer((0, 116))
            elif modifiers == 0:
                self._add_key_to_buffer((0, 80))
        elif symbol == key.RIGHT:
            if modifiers & key.MOD_CTRL:
                self._add_key_to_buffer((0, 116))
            elif modifiers == 0:
                self._add_key_to_buffer((0, 77))
        elif symbol <= 127:
            self._add_key_to_buffer(symbol)
        elif modifiers & key.MOD_CTRL:
            symbol = {
                key.NUM1: 1,
            }[symbol]
            self._add_key_to_buffer(symbol)

    def _add_key_to_buffer(self, key):
        self.keyboard_buf.append(key)
        if len(self.keyboard_buf) > KEYBOARD_BUF_SIZE:
            self.keyboard_buf = self.keyboard_buf[-KEYBOARD_BUF_SIZE:]

    def get_key(self):
        if self.keyboard_buf:
            return self.keyboard_buf.pop(0)
        else:
            return -1


def main():
    window = TerminalWindow(
        caption='QVM Terminal',
        resizable=True,
    )

    pyglet.app.run()


if __name__ == '__main__':
    main()
