import os

def encrypt(text):
    #Умножает код каждого символа на 2
    result = ""
    for char in text:
        result += chr(ord(char) * 2)
    return result

def decrypt(text):
    #Делит код каждого символа на 2
    result = ""
    for char in text:
        result += chr(ord(char) // 2)
    return result

def get_all_users():
    if not os.path.exists("user.txt"):
        return {}
        
    users_dict = {}
    with open("user.txt", "r", encoding="utf-8") as file:
        for line in file:
            if ":" in line:
                encrypted_login, encrypted_password = line.strip().split(":", 1)
                login = decrypt(encrypted_login)
                password = decrypt(encrypted_password)
                users_dict[login] = password
                
    return users_dict

def register(login, password):
    if login == "" or password == "":
        return False, "Поля не могут быть пустыми!"
        
    users_dict = get_all_users()
    if login in users_dict:
        return False, "Логин уже занят!"
    
    with open("user.txt", "a", encoding="utf-8") as file:
        file.write(f"{encrypt(login)}:{encrypt(password)}\n")
        
    return True, "Регистрация успешна!"

def login(login, password):
    users_dict = get_all_users()
    if login in users_dict and users_dict[login] == password:
        return True, "Вход выполнен!"
    else:
        return False, "Неверный логин или пароль!"