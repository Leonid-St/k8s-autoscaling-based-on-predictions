import logging
import smtplib
import imghdr
import os
import time
from email.message import EmailMessage
logger = logging.getLogger(__name__)

def send_email(email_receiver:str, deployment_name:str, text:str, pngfile:str, proactive):
    msg = EmailMessage()
    if proactive:
        type_text = "Proactive"
    else:
        type_text = "Reactive"
    msg['Subject'] = '{type_text} autoscaler for {deployment} notification'.format(type_text=type_text,
                                                                                   deployment=deployment_name)
    msg['From'] = "autoscaler"
    msg['To'] = email_receiver
    msg.set_content(text)

    with open(pngfile, 'rb') as fp:
        img_data = fp.read()
    msg.add_attachment(img_data, maintype='image', subtype=imghdr.what(None, img_data))

    for i in range(1,5):
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
                s.ehlo()
                s.login(os.environ["EMAIL_USERNAME"], os.environ["EMAIL_PASSWORD"])
                s.send_message(msg)
        except Exception as e:
            logger.error("Caught exception {} when trying to send an email, try {}/5".format(e, i+1))
            time.sleep(2)

    try:
        os.remove(pngfile)
    except OSError:
        pass
