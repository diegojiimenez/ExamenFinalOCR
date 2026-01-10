"""
Interfaz gráfica moderna y mejorada
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
from core.predictor import UniversalPredictor


class ModernOCRApp:
    """
    Aplicación GUI moderna con preview de imagen
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("🔤 OCR Inteligente - Sistema de Reconocimiento Universal")
        self.root.geometry("700x550")
        self.root.resizable(False, False)
        self.root.configure(bg='#1e1e1e')
        
        # Inicializar predictor
        try:
            self.predictor = UniversalPredictor()
            self.model_ready = True
        except FileNotFoundError as e:
            messagebox.showerror("Error", str(e))
            self.model_ready = False
        
        self.current_image_path = None
        self.setup_ui()
    
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        
        # === HEADER ===
        header_frame = tk.Frame(self.root, bg='#2d2d30', height=80)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title = tk.Label(
            header_frame, 
            text="🔤 OCR INTELIGENTE", 
            font=("Segoe UI", 22, "bold"),
            fg="#00d9ff", 
            bg='#2d2d30'
        )
        title.pack(pady=15)
        
        subtitle = tk.Label(
            header_frame, 
            text="Sistema de Reconocimiento Universal de Texto",
            font=("Segoe UI", 10),
            fg="#cccccc", 
            bg='#2d2d30'
        )
        subtitle.pack()
        
        # === MAIN CONTENT ===
        content_frame = tk.Frame(self.root, bg='#1e1e1e')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=30)
        
        # Panel de imagen
        self.image_panel = tk.Label(
            content_frame,
            text="📁 Ninguna imagen cargada",
            font=("Segoe UI", 11),
            fg="#888888",
            bg='#252526',
            width=60,
            height=10,
            relief=tk.SOLID,
            bd=1
        )
        self.image_panel.pack(pady=(0, 15))
        
        # Botones principales
        button_frame = tk.Frame(content_frame, bg='#1e1e1e')
        button_frame.pack(pady=10)
        
        self.btn_select = tk.Button(
            button_frame,
            text="📂 SELECCIONAR IMAGEN",
            command=self.select_image,
            bg='#0e639c',
            fg='white',
            font=("Segoe UI", 11, "bold"),
            width=22,
            height=2,
            relief=tk.FLAT,
            cursor='hand2'
        )
        self.btn_select.grid(row=0, column=0, padx=10)
        
        self.btn_recognize = tk.Button(
            button_frame,
            text="🎯 RECONOCER TEXTO",
            command=self.recognize_text,
            bg='#16825d',
            fg='white',
            font=("Segoe UI", 11, "bold"),
            width=22,
            height=2,
            relief=tk.FLAT,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.btn_recognize.grid(row=0, column=1, padx=10)
        
        # Panel de resultados
        result_frame = tk.LabelFrame(
            content_frame,
            text="📝 Resultado del Reconocimiento",
            font=("Segoe UI", 10, "bold"),
            fg="#00d9ff",
            bg='#252526',
            relief=tk.FLAT
        )
        result_frame.pack(fill=tk.BOTH, expand=True, pady=15)
        
        self.result_text = tk.Text(
            result_frame,
            height=4,
            font=("Consolas", 14),
            bg='#1e1e1e',
            fg='#d4d4d4',
            relief=tk.FLAT,
            padx=15,
            pady=15
        )
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # === FOOTER ===
        footer = tk.Label(
            self.root,
            text="💡 Condiciones óptimas: Fondo blanco, texto negro, letra imprenta, alto contraste",
            font=("Segoe UI", 8),
            fg="#666666",
            bg='#1e1e1e'
        )
        footer.pack(side=tk.BOTTOM, pady=10)
        
        # Estado del modelo
        status_text = "✅ Modelo cargado" if self.model_ready else "❌ Modelo no encontrado"
        status_color = "#16825d" if self.model_ready else "#e74856"
        
        status = tk.Label(
            self.root,
            text=status_text,
            font=("Segoe UI", 9),
            fg=status_color,
            bg='#1e1e1e'
        )
        status.pack(side=tk.BOTTOM)
    
    def select_image(self):
        """Selecciona una imagen para procesar"""
        filepath = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.bmp"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if filepath:
            self.current_image_path = filepath
            self.display_image(filepath)
            self.btn_recognize.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
    
    def display_image(self, filepath):
        """Muestra preview de la imagen"""
        try:
            img = Image.open(filepath)
            img.thumbnail((400, 200), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            self.image_panel.config(image=photo, text="")
            self.image_panel.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{str(e)}")
    
    def recognize_text(self):
        """Ejecuta el reconocimiento OCR"""
        if not self.current_image_path:
            return
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, "⏳ Procesando...")
        self.root.update()
        
        try:
            # Ejecutar predicción
            text, info = self.predictor.predict(self.current_image_path, debug=False)
            
            # Mostrar resultado
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(1.0, f"'{text}'")
            
            # Información adicional
            content_emojis = {"LETRA": "🔤", "PALABRA": "📝", "FRASE": "📄"}
            emoji = content_emojis.get(info['type'], "📋")
            
            messagebox.showinfo(
                "Reconocimiento Exitoso",
                f"{emoji} Tipo: {info['type']}\n"
                f"🔍 Caracteres: {info['num_chars']}\n"
                f"📝 Texto: '{text}'"
            )
            
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(1.0, "❌ Error en el reconocimiento")
            messagebox.showerror("Error", f"Error durante el reconocimiento:\n{str(e)}")


def launch_gui():
    """Lanza la interfaz gráfica"""
    root = tk.Tk()
    app = ModernOCRApp(root)
    root.mainloop()