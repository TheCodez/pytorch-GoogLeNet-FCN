from sinchsms import SinchSMS


class SMSHandler(object):

    def __init__(self, credentials):
        key, secret, number = credentials.split(' ')

        self.client = SinchSMS(key, secret)
        self.number = number

    def __call__(self, message):
        self.client.send_message(self.number, message)
