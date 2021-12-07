from tutils.templates.simple_script_helper import template

if __name__ == '__main__':
    print("outside", __file__)
    dictt = template(__file__)
    print(dictt)