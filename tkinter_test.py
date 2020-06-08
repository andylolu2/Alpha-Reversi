from tkinter import *


def enter(event):
    global end
    end.set(False)


master = Tk()
dimension = 800
end = BooleanVar()
end.set(True)
tk = Canvas(master, width=dimension, height=dimension)
center = dimension / 2
width = 230
height = 90
tk.create_rectangle(center - width / 2, center - height / 2, center + width / 2, center + height / 2,
                    fill="#2d3445", outline="#2d3445")
tk.create_text(dimension / 2, dimension / 2, text="White Won!\nWhite 30 : 34 Black\nPress Enter to exit", font=("Arial", 15),fill="#fcfdff", justify=CENTER)
tk.bind('<Return>', enter)
tk.focus_set()
tk.pack()
tk.update()

while True:
    tk.wait_variable(end)
    print("Finish")
    break
