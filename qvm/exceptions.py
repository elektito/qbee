class DeviceError(Exception):
    def __init__(self, error_msg, error_code=None):
        self.error_msg = error_msg
        self.error_code = error_code

    def __str__(self):
        return self.error_msg

    def __repr__(self):
        return (
            f'<DeviceError code={self.error_code} msg={self.error_msg}>'
        )
