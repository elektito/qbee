class PrintUsingFormatter:
    """Formats text according to QBASIC's PRINT USING statement rules.

    """

    def __init__(self, fmt):
        self.fmt = fmt
        self.fmt_parts = []
        self.parse_format_string(fmt)


    def parse_format_string(self, fmt):
        i = 0
        non_formatting = ''
        redo_no_number = False
        while i < len(fmt):
            if fmt[i] in ['#', '+', '-'] and not redo_no_number:
                if non_formatting:
                    self.fmt_parts.append(('non', non_formatting))
                    non_formatting = ''
                n, part = self.parse_numeric_format_string(fmt, i)
                if part[2]['real_sharps'] == 0:
                    redo_no_number = True
                    continue
                else:
                    i += n
                    self.fmt_parts.append(part)

            if i >= len(fmt):
                break

            redo_no_number = False
            if fmt[i] in ['&', '!']:
                if non_formatting:
                    self.fmt_parts.append(('non', non_formatting))
                    non_formatting = ''
                self.fmt_parts.append(('str', fmt[i]))
                i += 1
            elif fmt[i] == '_':
                non_formatting += fmt[i+1]
                i += 2
            else:
                non_formatting += fmt[i]
                i += 1

        if non_formatting:
            self.fmt_parts.append(('non', non_formatting))


    def parse_numeric_format_string(self, fmt, idx):
        options = {}
        sign = ''
        i = idx
        sharps = 0
        real_sharps = 0

        if fmt[i] in '+-':
            options['sign'] = ('begin', fmt[i])
            sign = fmt[i]
            sharps += 1
            i += 1

        while i < len(fmt):
            if not sign and fmt[i] in '+-':
                options['sign'] = ('end', fmt[i])
                sharps += 1
                i += 1
                break
            elif fmt[i] == '#':
                sharps += 1
                real_sharps += 1
                i += 1
            elif fmt[i] == ',':
                options['comma'] = True
                sharps += 1
                i += 1
            elif fmt[i] == '.':
                if 'decimal_point' in options:
                    break
                sharps += 1
                options['decimal_point'] = sharps
                i += 1
            else:
                break

        options['real_sharps'] = real_sharps
        return i - idx, ('num', sharps*'#', options)


    def format(self, values):
        fmt_parts = [
            p if len(p) == 3 else (p + ({},))
            for p in self.fmt_parts
        ]
        output = ''
        i = 0
        for (fmt_type, fmt, options) in fmt_parts:
            if fmt_type != 'non' and i >= len(fmt_parts):
                raise RuntimeError('Not enough values.')

            if fmt_type == 'non':
                output += fmt
            elif fmt_type == 'str':
                if not isinstance(values[i], str):
                    raise RuntimeError('Type mismatch.')
                output += values[i][0] if fmt == '!' else values[i]
                i += 1
            elif fmt_type == 'num':
                output += self.format_number(fmt, values[i], options)
                i += 1
            else:
                assert False

        if i < len(values):
            raise RuntimeError('Too many values.')

        return output


    def format_number(self, fmt, value, options):
        fmt_str = '{:'
        if options.get('comma', False):
            fmt_str += ','
        if 'decimal_point' in options:
            fmt_str += '.'
            fmt_str += str(len(fmt) - options['decimal_point'])
            fmt_str += 'f'
        fmt_str += '}'

        if 'sign' in options:
            sign_pos, sign_type = options['sign']
        else:
            sign_pos, sign_type = 'begin', '-'

        sign = -1 if value < 0 else 1
        value = abs(value)

        result = fmt_str.format(value)

        if sign_type == '-':
            sign = '-' if sign == -1 else ' '
        else:
            sign = '-' if sign == -1 else '+'

        if sign_pos == 'begin':
            result = sign + result
        else:
            result = result + sign
            if sign != '-':
                result = ' ' + result

        if len(result) < len(fmt):
            result = ' ' * (len(fmt) - len(result)) + result

        if sign == ' ' and len(result) > len(fmt) and sign_pos == 'begin':
            result = result[1:]
        elif sign == ' ' and len(result) > len(fmt) and sign_pos == 'end':
            result = result[:-1]

        if len(result) > len(fmt):
            result = '%' + result

        return result
