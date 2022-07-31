import signal
import pyglet
import numpy as np
from queue import Empty
from functools import lru_cache
from PIL import Image, ImageDraw
from .exceptions import DeviceError


KEYBOARD_BUF_SIZE = 8


# Resource paths are relative to where to python module is
pyglet.resource.path = ['..']
pyglet.resource.reindex()


class TerminalWindow(pyglet.window.Window):
    def __init__(self, request_queue, result_queue,
                 ignore_keyboard_interrupt=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if ignore_keyboard_interrupt:
            signal.signal(signal.SIGINT, lambda a, b: None)

        self.set_mode(0)

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

        # set current background and foreground color for the entire
        # screen text buffer
        attrs = self._get_current_text_attrs_byte()
        for i in range(1, len(self.text_buffer), 2):
            self.text_buffer[i] = attrs

        self._cursor_row = self._cursor_col = 0

    def set_mode(self, mode):
        assert mode == 0, 'Only SCREEN 0 supported for now'
        self.mode = mode

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

        # by default the "view print" area, spans the entire screen
        # except for the last line. this seems to be the default in
        # QBASIC (or DOS?) too, because if you print something on the
        # last line, the rest of the screen scrolls, but last line is
        # not affected. If you perform a single "VIEW PRINT" command
        # (with no arguments, meaning the entire screen), then writing
        # to the last line causes scroll.
        self.text_view_top = 0
        self.text_view_bottom = self.text_lines - 2

        # cursor position on the screen
        self._cursor_row = self._cursor_col = 0

    def locate(self, row, col):
        if row is not None:
            if row < self.text_view_top or \
               row > self.text_view_bottom:
                # for some reason QB allows moving the cursor to the
                # bottom of the screen, even if its out of VIEW PRINT
                # area.
                if row != self.text_lines - 1:
                    raise DeviceError(error_msg='Invalid cursor row')
            self._cursor_row = row

        if col is not None:
            if col < 0 or col >= self.text_columns:
                raise DeviceError('Invalid cursor col')

            self._cursor_col = col

    def get_cursor_pos(self):
        return self._cursor_row, self._cursor_col

    def view_print(self, top_line, bottom_line):
        if top_line is None:
            top_line = 0
        if bottom_line is None:
            bottom_line = self.text_lines - 1

        self.text_view_top = top_line
        self.text_view_bottom = bottom_line

    def put_text(self, text):
        if isinstance(text, str):
            text = text.encode('cp437')
        assert isinstance(text, (bytes, bytearray))
        for char in text:
            # the following check is performed here, instead of at the
            # bottom of the loop where we actually increment
            # _cursor_col, because we don't want any possible scrolls
            # immediately happen, but on the next time we want to
            # write something to the screen.
            #
            # this seems to be consistent with how QB itself works. If
            # you perform a `PRINT "x";` at the bottom-right corner of
            # the screen, the screen won't scroll, even though
            # technically the cursor is now outside the
            # screen. Instead, next time you try to write something,
            # the scroll is performed.
            #
            # notice that this might leave _cursor_col outside valid
            # range when this function returns.
            if self._cursor_col == self.text_columns:
                self._cursor_col = 0
                self._cursor_row += 1

                self.scroll_if_necessary()

            idx = self._cursor_row * self.text_columns + self._cursor_col
            idx *= 2

            if idx > len(self.text_buffer) - 2:
                break

            if char == 13:
                self._cursor_col = 0
                continue

            if char == 10:
                self._cursor_row += 1
                self.scroll_if_necessary()
                continue

            attrs = self._get_current_text_attrs_byte()
            self.text_buffer[idx] = char
            self.text_buffer[idx + 1] = attrs

            self._cursor_col += 1

        self._text_updated = True

    def scroll_if_necessary(self):
        # this function can be called after the cursor has just been
        # moved to the next line after printing one character to the
        # screen. depending on the position of the cursor (which might
        # be temporarily in an invalid position), it will perform
        # screen scrolling if necessary. The cursor position is also
        # fixed.

        if self._cursor_row > self.text_view_bottom:
            # last line will not scroll itself, but moving past it
            # _will_ cause a scroll (of the rest of the screen)
            self.scroll()
            self._cursor_row = self.text_view_bottom

    def scroll(self):
        # scroll the entire screen except for the last line
        start = self.text_columns * 2
        end = -self.text_columns * 2

        empty_char = bytes([self._get_current_text_attrs_byte(), 0])
        empty_line = bytearray(empty_char * self.text_columns)

        last_line = self.text_buffer[end:]
        scroll_area = self.text_buffer[start:end]
        self.text_buffer = scroll_area + empty_line + last_line

    def update_text(self):
        draw = ImageDraw.Draw(self.text_img)
        draw.rectangle([0, 0, self.text_width, self.text_height],
                       fill=self.bg_color)

        for i in range(0, len(self.text_buffer), 2):
            line = (i // 2) // self.text_columns
            column = (i // 2) % self.text_columns
            x = column * self.char_width
            y = line * self.char_height

            attrs = self.text_buffer[i + 1]
            fg_color = attrs & 0x0f
            bg_color = (attrs & 0x70) >> 4
            blink = (attrs & 0x80) >> 7
            fg_color = self.color_palette[fg_color]
            bg_color = self.color_palette[bg_color]
            char_code = self.text_buffer[i]
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

    def set_mem(self, offset, value):
        self.text_buffer[offset] = value
        self._text_updated = True

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

    def _get_current_text_attrs_byte(self):
        fg_color = self.fg_color
        bg_color = self.bg_color
        blink = 0x00
        if fg_color >= 16:
            fg_color -= 16
            blink = 0x80
        return blink | (bg_color << 4) | fg_color

    def on_draw(self):
        if self._text_updated:
            self.update_text()
            self._text_updated = False
        self.clear()
        self.text_texture.blit(
            0, 0, width=self.width, height=self.height)

        if self.show_cursor and \
           0 <= (self.time % 1.0) <= self.cursor_blink_show_time:
            text_bottom = self.text_texture.height - 1
            x = self._cursor_col * self.char_width
            y = text_bottom - (self._cursor_row * self.char_height + self.cursor_start)
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
             modifiers & key.MOD_CTRL:
            self._add_key_to_buffer(symbol - ord('a') + 1)
        elif (ord('[') <= symbol <= ord('_')) and \
             modifiers & key.MOD_CTRL:
            self._add_key_to_buffer(symbol - ord('[') + 27)
        elif symbol == ord('6') and modifiers & key.MOD_CTRL:
            self._add_key_to_buffer(30)
        elif symbol == ord('-') and modifiers & key.MOD_CTRL:
            self._add_key_to_buffer(31)
        elif (ord('a') <= symbol <= ord('z')) and \
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
