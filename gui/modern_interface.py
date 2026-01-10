"""
GUI con visualización estilo Examen Final
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('TkAgg')

from core.predictor import UniversalPredictor


class ModernOCRApp:
    """GUI con visualización integrada"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🔤 OCR Inteligente - Sistema de Reconocimiento Universal")
        self.root.geometry("900x650")
        self.root.configure(bg='#1e1e1e')
        
        try:
            self.predictor = UniversalPredictor()
            self.model_ready = True
        except:
            self.model_ready = False
            self.predictor = None
        
        self.current_image_path = None
        self.setup_ui()
    
    def setup_ui(self):
        """Configurar interfaz"""
        
        # Header
        header = tk.Frame(self.root, bg='#2d2d30', height=80)
        header.pack(fill=tk.X)
        
        tk.Label(
            header, text="🔤 OCR INTELIGENTE",
            font=("Segoe UI", 22, "bold"),
            fg="#00d9ff", bg='#2d2d30'
        ).pack(pady=15)
        
        tk.Label(
            header, text="Sistema de Reconocimiento Universal de Texto",
            font=("Segoe UI", 10),
            fg="#cccccc", bg='#2d2d30'
        ).pack()
        
        # Content
        content = tk.Frame(self.root, bg='#1e1e1e')
        content.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)
        
        # Preview
        self.image_panel = tk.Label(
            content, text="📁 Ninguna imagen cargada",
            font=("Segoe UI", 11), fg="#888888",
            bg='#252526', width=70, height=12,
            relief=tk.SOLID, bd=1
        )
        self.image_panel.pack(pady=(0, 15))
        
        # Botones
        btn_frame = tk.Frame(content, bg='#1e1e1e')
        btn_frame.pack(pady=10)
        
        tk.Button(
            btn_frame, text="📂 SELECCIONAR",
            command=self.select_image,
            bg='#0e639c', fg='white',
            font=("Segoe UI", 11, "bold"),
            width=18, height=2,
            relief=tk.FLAT, cursor='hand2'
        ).grid(row=0, column=0, padx=5)
        
        tk.Button(
            btn_frame, text="🎯 RECONOCER",
            command=self.recognize_text,
            bg='#16825d', fg='white',
            font=("Segoe UI", 11, "bold"),
            width=18, height=2,
            relief=tk.FLAT, cursor='hand2',
            state=tk.NORMAL if self.model_ready else tk.DISABLED
        ).grid(row=0, column=1, padx=5)
        
        tk.Button(
            btn_frame, text="🔍 VISUALIZAR",
            command=self.show_visualization,
            bg='#8e44ad', fg='white',
            font=("Segoe UI", 11, "bold"),
            width=18, height=2,
            relief=tk.FLAT, cursor='hand2',
            state=tk.NORMAL if self.model_ready else tk.DISABLED
        ).grid(row=0, column=2, padx=5)
        
        # Resultado
        result_frame = tk.LabelFrame(
            content, text="📝 Resultado",
            font=("Segoe UI", 10, "bold"),
            fg="#00d9ff", bg='#252526',
            relief=tk.FLAT
        )
        result_frame.pack(fill=tk.X, pady=15)
        
        self.result_text = tk.Text(
            result_frame, height=4,
            font=("Consolas", 14),
            bg='#1e1e1e', fg='#d4d4d4',
            relief=tk.FLAT, padx=15, pady=15
        )
        self.result_text.pack(fill=tk.X, padx=10, pady=10)
        
        # Footer
        status_text = "✅ Modelo listo" if self.model_ready else "❌ Sin modelo"
        status_color = "#16825d" if self.model_ready else "#e74856"
        
        tk.Label(
            self.root, text=status_text,
            font=("Segoe UI", 9),
            fg=status_color, bg='#1e1e1e'
        ).pack(side=tk.BOTTOM, pady=10)
        
        tk.Label(
            self.root,
            text="💡 Óptimo: fondo blanco, texto negro, imprenta, alto contraste",
            font=("Segoe UI", 8),
            fg="#666666", bg='#1e1e1e'
        ).pack(side=tk.BOTTOM, pady=5)
    
    def select_image(self):
        """Seleccionar imagen"""
        filepath = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.bmp"),
                ("Todos", "*.*")
            ]
        )
        
        if filepath:
            self.current_image_path = filepath
            self.display_image(filepath)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(1.0, f"Imagen: {filepath}")
    
    def display_image(self, filepath):
        """Mostrar preview"""
        try:
            img = Image.open(filepath)
            img.thumbnail((600, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            self.image_panel.config(image=photo, text="")
            self.image_panel.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar:\n{e}")
    
    def recognize_text(self):
        """Reconocer sin visualización"""
        if not self.current_image_path or not self.model_ready:
            messagebox.showwarning("Advertencia", "Selecciona imagen")
            return
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, "⏳ Procesando...")
        self.root.update()
        
        try:
            text, info = self.predictor.predict(
                self.current_image_path, debug=False
            )
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(1.0, f"'{text}'")
            
            emoji = {"LETRA": "🔤", "PALABRA": "📝", "FRASE": "📄"}.get(info['type'], "📋")
            
            messagebox.showinfo(
                "Resultado",
                f"{emoji} Tipo: {info['type']}\n"
                f"🔍 Caracteres: {info['num_chars']}\n"
                f"📝 Texto: '{text}'"
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def show_visualization(self):
        """Mostrar visualización completa"""
        if not self.current_image_path or not self.model_ready:
            messagebox.showwarning("Advertencia", "Selecciona imagen")
            return
        
        try:
            text, info = self.predictor.show_visualization(
                self.current_image_path
            )
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(1.0, f"'{text}'")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))


def launch_gui():
    """Lanzar GUI"""
    root = tk.Tk()
    app = ModernOCRApp(root)
    root.mainloop()