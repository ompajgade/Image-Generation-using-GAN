import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import torch
import numpy as np
from generator_1 import Generator  # Import the generator class
import os

# Load the pre-trained generator model
G = Generator().cuda()
G.load_state_dict(torch.load('generator.pth'))
G.eval()

# Initialize the Tkinter window
root = tk.Tk()
root.title("Image Generator")
root.geometry("600x600")

# Set the background image
bg_image = Image.open("background_2.jpg")  # Ensure your background image is set correctly here
bg_image = bg_image.resize((600, 600), Image.Resampling.LANCZOS)
bg_image_tk = ImageTk.PhotoImage(bg_image)

background_label = tk.Label(root, image=bg_image_tk)
background_label.place(relwidth=1, relheight=1)

# Create a frame to contain the content without background (or matching background color)
frame = tk.Frame(root, bg="#F0F0F0", bd=0)  # Use a matching color to your background image
frame.place(relx=0.5, rely=0.1, relwidth=0.75, relheight=0.8, anchor='n')

# Title label with custom font and no background
title_label = tk.Label(frame, text="Image Generator", font=("Helvetica", 24, "bold"), fg="black", bg="#F0F0F0")
title_label.pack(pady=10)

# Image display area with a white border outline and background color matching
img_frame = tk.Frame(frame, bg="#F0F0F0", highlightbackground="white", highlightthickness=2)
img_frame.pack(pady=10)

img_label = tk.Label(img_frame, bg="#F0F0F0")  # Matching background
img_label.pack()

# Variable to hold the generated image for saving
generated_image = None

# Function to generate and display the image
def generate_image():
    global generated_image
    z = torch.randn(1, 100).cuda()  # Random noise
    fake_img = G(z).cpu().detach().numpy().reshape(28, 28)

    # Convert image to 0-255 range
    img = (fake_img * 127.5 + 127.5).astype(np.uint8)
    img = Image.fromarray(img)
    img = img.resize((200, 200), Image.Resampling.LANCZOS)  # Resize to display

    # Display the image in the label
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk

    # Save the image for saving later
    generated_image = img

# Function to save the generated image
def save_image():
    global generated_image
    if generated_image:
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if save_path:
            generated_image.save(save_path)
            print(f"Image saved as {save_path}")

# Button hover effect for Generate Image button
def on_generate_hover(e):
    generate_btn['background'] = '#4CAF50'
    generate_btn['foreground'] = 'white'

def on_generate_leave(e):
    generate_btn['background'] = 'white'
    generate_btn['foreground'] = 'black'

# Button hover effect for Save Image button
def on_save_hover(e):
    save_btn['background'] = '#FFA500'
    save_btn['foreground'] = 'white'

def on_save_leave(e):
    save_btn['background'] = 'white'
    save_btn['foreground'] = 'black'

# Generate Image button with hover effects
generate_btn = tk.Button(frame, text="Generate Image", font=("Helvetica", 16), padx=20, pady=10, fg="black", bg="white", command=generate_image)
generate_btn.bind("<Enter>", on_generate_hover)
generate_btn.bind("<Leave>", on_generate_leave)
generate_btn.pack(pady=10)

# Save Image button with hover effects
save_btn = tk.Button(frame, text="Save Image", font=("Helvetica", 16), padx=20, pady=10, fg="black", bg="white", command=save_image)
save_btn.bind("<Enter>", on_save_hover)
save_btn.bind("<Leave>", on_save_leave)
save_btn.pack(pady=10)

# Additional window styling
style = ttk.Style()
style.configure("TButton", padding=6, relief="flat", background="#ccc")

# Run the Tkinter event loop
root.mainloop()