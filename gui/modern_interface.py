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
        self.root.title("🔤 OCR Inteligente - Sistema de Reconocimiento de Texto")
        self.root.geometry("1000x700")
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
        header = tk.Frame(self.root, bg='#2d2d30', height=100)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(
            header, text="🔤 OCR INTELIGENTE",
            font=("Segoe UI", 24, "bold"),
            fg="#00d9ff", bg='#2d2d30'
        ).pack(pady=(20, 5))
        
        tk.Label(
            header, text="Sistema de Reconocimiento Universal de Texto",
            font=("Segoe UI", 11),
            fg="#cccccc", bg='#2d2d30'
        ).pack()
        
        # Content
        content = tk.Frame(self.root, bg='#1e1e1e')
        content.pack(fill=tk.BOTH, expand=True, padx=40, pady=30)
        
        # Preview Frame (con scroll si es necesario)
        preview_frame = tk.LabelFrame(
            content, text="📷 Vista Previa",
            font=("Segoe UI", 11, "bold"),
            fg="#00d9ff", bg='#252526',
            relief=tk.FLAT, bd=2
        )
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Canvas para imagen con mejor control
        self.canvas = tk.Canvas(
            preview_frame,
            bg='#1e1e1e',
            highlightthickness=0,
            cursor='hand2'
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Texto placeholder
        self.placeholder_text = self.canvas.create_text(
            400, 200,
            text="📁 Ninguna imagen cargada\nHaz clic en 'SELECCIONAR' para comenzar",
            font=("Segoe UI", 12),
            fill="#888888",
            justify=tk.CENTER
        )
        
        # Botones
        btn_frame = tk.Frame(content, bg='#1e1e1e')
        btn_frame.pack(pady=15)
        
        tk.Button(
            btn_frame, text="📂 SELECCIONAR",
            command=self.select_image,
            bg='#0e639c', fg='white',
            font=("Segoe UI", 12, "bold"),
            width=16, height=2,
            relief=tk.FLAT, cursor='hand2',
            activebackground='#1177bb'
        ).grid(row=0, column=0, padx=8)
        
        tk.Button(
            btn_frame, text="🎯 RECONOCER",
            command=self.recognize_text,
            bg='#16825d', fg='white',
            font=("Segoe UI", 12, "bold"),
            width=16, height=2,
            relief=tk.FLAT, cursor='hand2',
            activebackground='#1a9970',
            state=tk.NORMAL if self.model_ready else tk.DISABLED
        ).grid(row=0, column=1, padx=8)
        
        tk.Button(
            btn_frame, text="🔍 VISUALIZAR",
            command=self.show_visualization,
            bg='#8e44ad', fg='white',
            font=("Segoe UI", 12, "bold"),
            width=16, height=2,
            relief=tk.FLAT, cursor='hand2',
            activebackground='#9b59b6',
            state=tk.NORMAL if self.model_ready else tk.DISABLED
        ).grid(row=0, column=2, padx=8)
        
        # Resultado
        result_frame = tk.LabelFrame(
            content, text="📝 Texto Reconocido",
            font=("Segoe UI", 11, "bold"),
            fg="#00d9ff", bg='#252526',
            relief=tk.FLAT, bd=2
        )
        result_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.result_text = tk.Text(
            result_frame, height=3,
            font=("Consolas", 15, "bold"),
            bg='#1e1e1e', fg='#4ec9b0',
            relief=tk.FLAT, padx=20, pady=15,
            wrap=tk.WORD
        )
        self.result_text.pack(fill=tk.X, padx=15, pady=15)
        
        # Footer
        footer = tk.Frame(self.root, bg='#2d2d30', height=40)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)
        
        status_text = "✅ Modelo cargado y listo" if self.model_ready else "❌ Modelo no disponible"
        status_color = "#16825d" if self.model_ready else "#e74856"
        
        tk.Label(
            footer, text=status_text,
            font=("Segoe UI", 10),
            fg=status_color, bg='#2d2d30'
        ).pack(pady=10)
    
    def select_image(self):
        """Seleccionar imagen"""
        filepath = filedialog.askopenfilename(
            title="Seleccionar imagen para OCR",
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("Todos", "*.*")
            ]
        )
        
        if filepath:
            self.current_image_path = filepath
            self.display_image(filepath)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(1.0, f"📁 Archivo: {filepath.split('/')[-1]}")
    
    def display_image(self, filepath):
        """Mostrar preview con mejor escalado"""
        try:
            # Limpiar canvas
            self.canvas.delete("all")
            
            # Cargar imagen
            img = Image.open(filepath)
            
            # Obtener dimensiones del canvas
            self.canvas.update()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Calcular escala manteniendo aspect ratio
            img_width, img_height = img.size
            scale_w = canvas_width / img_width
            scale_h = canvas_height / img_height
            scale = min(scale_w, scale_h, 1.0) * 0.9  # 90% del espacio disponible
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Redimensionar con alta calidad
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convertir a PhotoImage
            self.photo = ImageTk.PhotoImage(img_resized)
            
            # Centrar en canvas
            x = canvas_width // 2
            y = canvas_height // 2
            
            self.canvas.create_image(x, y, image=self.photo, anchor=tk.CENTER)
            
        except Exception as e:
            messagebox.showerror("Error al cargar imagen", f"No se pudo cargar la imagen:\n{e}")
            self.canvas.create_text(
                400, 200,
                text=f"❌ Error al cargar imagen\n{str(e)[:100]}",
                font=("Segoe UI", 11),
                fill="#e74856",
                justify=tk.CENTER
            )
    
    def recognize_text(self):
        """Reconocer sin visualización"""
        if not self.current_image_path or not self.model_ready:
            messagebox.showwarning("Advertencia", "Por favor selecciona una imagen primero")
            return
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, "⏳ Procesando imagen...")
        self.root.update()
        
        try:
            text, info = self.predictor.predict(
                self.current_image_path, debug=False
            )
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(1.0, f"✨ {text}")
            
            emoji = {"LETRA": "🔤", "PALABRA": "📝", "FRASE": "📄"}.get(info['type'], "📋")
            
            messagebox.showinfo(
                "✅ Reconocimiento Exitoso",
                f"{emoji} Tipo detectado: {info['type']}\n"
                f"🔍 Total de caracteres: {info['num_chars']}\n"
                f"📝 Texto reconocido:\n\n'{text}'"
            )
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(1.0, f"❌ Error: {str(e)[:100]}")
            messagebox.showerror("Error", f"No se pudo reconocer el texto:\n{e}")
    
    def show_visualization(self):
        """Mostrar visualización completa"""
        if not self.current_image_path or not self.model_ready:
            messagebox.showwarning("Advertencia", "Por favor selecciona una imagen primero")
            return
        
        try:
            text, info = self.predictor.show_visualization(
                self.current_image_path
            )
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(1.0, f"✨ {text}")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo generar la visualización:\n{e}")


def launch_gui():
    """Lanzar GUI"""
    root = tk.Tk()
    app = ModernOCRApp(root)
    root.mainloop()