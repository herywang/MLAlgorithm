
def get_file_content(file='./__init__.py'):
    key = 'yuteryxs'
    i = 0
    len_key = len(key)
    new_file = open("./new_file", 'wb')
    with open(file, 'r') as f:
        str = f.read()
        print(len(str))
        new_str = ""
        for ch in str:
            if i == len_key:
                i = 0
            # print('original: %d'%ord(ch))
            new_ch = chr(ord(ch) + ord(key[i]) - 120)
            # print('new char : %d'%ord(new_ch))
            i += 1
            new_str += new_ch
        print(new_str)
        for ch in new_str:
            print(ord(ch))
        print(len(new_str))
        new_file.write(new_str.encode('utf-8'))
    new_file.close()

def unlock_file():
    key = 'yuteryxs'
    i = 0
    len_key = len(key)
    unlock_file_text = ""
    unl_file = open('./unlocak_file', 'w')
    with open('./new_file', 'r') as f:
        str = f.read()
        print(str)
        for ch in str:
            print(ord(ch))
            if i == len_key:
                i = 0
            unlock_file_text += chr(ord(ch)  - ord(key[i]) + 120)
            i += 1
        for ch in unlock_file_text:
            print(ord(ch))
        print(len(unlock_file_text))
        unl_file.write(unlock_file_text)

unlock_file()
# get_file_content()

