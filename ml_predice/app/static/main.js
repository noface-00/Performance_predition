/* ===================================
   Sistema de Predicción Estudiantil
   Archivo JavaScript Principal
   =================================== */

// ===== VARIABLES GLOBALES =====
let isFormSubmitting = false;
let uploadedFile = null;

// ===== INICIALIZACIÓN =====
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Sistema de Predicción Estudiantil cargado');
    
    initializeFormHandlers();
    initializeUploadHandlers();
    initializeAnimations();
    initializeValidations();
    
    // Mostrar mensaje de bienvenida
    showWelcomeMessage();
});

// ===== MANEJO DE FORMULARIOS =====
function initializeFormHandlers() {
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', handlePredictionSubmit);
    }
    
    // Auto-sugerencias inteligentes
    const edadInput = document.getElementById('edad');
    if (edadInput) {
        edadInput.addEventListener('change', handleEdadChange);
    }
    
    const tipoActividadSelect = document.getElementById('tipo_actividad');
    if (tipoActividadSelect) {
        tipoActividadSelect.addEventListener('change', handleTipoActividadChange);
    }
}

function handlePredictionSubmit(e) {
    if (isFormSubmitting) {
        e.preventDefault();
        return false;
    }
    
    const submitBtn = document.querySelector('.btn-predict');
    if (submitBtn) {
        isFormSubmitting = true;
        
        // Cambiar apariencia del botón
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analizando...';
        submitBtn.disabled = true;
        submitBtn.classList.add('loading');
        
        // Mostrar animación de carga
        showLoadingAnimation();
        
        // Restaurar botón después del envío
        setTimeout(() => {
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
            submitBtn.classList.remove('loading');
            isFormSubmitting = false;
        }, 1000);
    }
}

function handleEdadChange() {
    const edad = parseInt(this.value);
    const gradoField = document.getElementById('grado');
    
    if (edad >= 6 && edad <= 18 && gradoField && !gradoField.value) {
        // Sugerir grado basado en edad
        const suggestedGrade = Math.max(1, Math.min(12, edad - 5));
        gradoField.value = suggestedGrade;
        
        // Efecto visual
        highlightField(gradoField, '#e8f5e8');
        
        // Mostrar tooltip
        showTooltip(gradoField, `Grado sugerido: ${suggestedGrade}° basado en edad ${edad}`);
    }
}

function handleTipoActividadChange() {
    const dificultadField = document.getElementById('dificultad');
    const activityType = this.value;
    
    if (!dificultadField) return;
    
    // Sugerencias de dificultad por tipo de actividad
    const suggestions = {
        'sopa_letras': 2,
        'crucigrama': 3,
        'relacionar': 2,
        'memoria': 1
    };
    
    if (suggestions[activityType]) {
        dificultadField.value = suggestions[activityType];
        highlightField(dificultadField, '#e8f5e8');
        
        const difficultyNames = ['', 'Básico', 'Intermedio', 'Avanzado', 'Experto'];
        showTooltip(dificultadField, `Dificultad sugerida: ${difficultyNames[suggestions[activityType]]}`);
    }
}

// ===== MANEJO DE UPLOAD =====
function initializeUploadHandlers() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('csvFile');
    const uploadBtn = document.getElementById('uploadBtn');
    
    if (!uploadArea || !fileInput) return;
    
    // Click para seleccionar archivo
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Cambio de archivo
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    // Botón de upload
    if (uploadBtn) {
        uploadBtn.addEventListener('click', uploadCSV);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    this.classList.add('dragover');
}

function handleDragLeave() {
    this.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    this.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (file.name.toLowerCase().endsWith('.csv')) {
            const fileInput = document.getElementById('csvFile');
            if (fileInput) {
                fileInput.files = files;
                handleFileSelect(file);
            }
        } else {
            showAlert('error', 'Por favor, selecciona un archivo CSV válido.');
        }
    }
}

function handleFileSelect(file) {
    uploadedFile = file;
    
    // Actualizar información del archivo
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const fileInfo = document.getElementById('fileInfo');
    const uploadArea = document.getElementById('uploadArea');
    const uploadBtn = document.getElementById('uploadBtn');
    
    if (fileName) fileName.textContent = file.name;
    if (fileSize) fileSize.textContent = formatFileSize(file.size);
    if (fileInfo) fileInfo.classList.add('show');
    
    // Actualizar apariencia del área de upload
    if (uploadArea) {
        uploadArea.classList.add('file-selected');
        uploadArea.innerHTML = `
            <div class="upload-icon">
                <i class="fas fa-file-csv"></i>
            </div>
            <div class="upload-text">Archivo seleccionado: ${file.name}</div>
            <div class="upload-subtext">Listo para subir y reentrenar</div>
        `;
    }
    
    // Habilitar botón de upload
    if (uploadBtn) {
        uploadBtn.classList.add('enabled');
        uploadBtn.disabled = false;
    }
    
    // Animación de confirmación
    playSuccessAnimation(uploadArea);
}

async function uploadCSV() {
    const fileInput = document.getElementById('csvFile');
    if (!fileInput || fileInput.files.length === 0) {
        showAlert('error', 'Por favor, selecciona un archivo CSV');
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    const progressSection = document.getElementById('progressSection');
    const outputSection = document.getElementById('outputSection');
    const output = document.getElementById('output');
    const uploadBtn = document.getElementById('uploadBtn');
    
    try {
        // Mostrar secciones de progreso
        if (progressSection) progressSection.style.display = 'block';
        if (outputSection) outputSection.style.display = 'block';
        
        // Deshabilitar botón
        if (uploadBtn) {
            uploadBtn.disabled = true;
            uploadBtn.innerHTML = '<div class="spinner"></div> Procesando...';
        }

        // Paso 1: Subir archivo
        updateProgress(25, 'Subiendo archivo CSV...');
        
        const uploadResp = await fetch('/upload_csv', {
            method: 'POST',
            body: formData
        });
        
        const uploadData = await uploadResp.json();
        
        if (output) {
            output.textContent = JSON.stringify(uploadData, null, 2);
        }
        
        if (uploadData.error) {
            throw new Error(uploadData.error);
        }

        // Paso 2: Reentrenar modelo
        updateProgress(50, 'Iniciando reentrenamiento...');
        
        const retrainResp = await fetch('/retrain', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ filepath: uploadData.ruta })
        });
        
        const retrainData = await retrainResp.json();
        
        updateProgress(75, 'Validando modelo...');
        
        // Completar
        updateProgress(100, 'Reentrenamiento completado');
        
        if (output) {
            output.textContent += '\n\n' + JSON.stringify(retrainData, null, 2);
        }
        
        if (retrainData.error) {
            throw new Error(retrainData.error);
        }
        
        // Éxito
        updateStatusBadge('success', 'Completado exitosamente');
        
        // Mostrar mensaje de éxito
        setTimeout(() => {
            if (confirm('¡Modelo reentrenado exitosamente! ¿Deseas volver al predictor?')) {
                window.location.href = '/';
            }
        }, 2000);
        
    } catch (error) {
        console.error('Error en upload:', error);
        
        // Manejo de errores
        updateStatusBadge('error', 'Error en el proceso');
        
        if (output) {
            output.textContent += '\n\nError: ' + error.message;
        }
        
        updateProgress(0, 'Error en el reentrenamiento');
        const progressFill = document.getElementById('progressFill');
        if (progressFill) {
            progressFill.style.background = 'var(--danger-gradient)';
        }
        
        showAlert('error', 'Error durante el reentrenamiento: ' + error.message);
    }
    
    // Rehabilitar botón
    if (uploadBtn) {
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = '<i class="fas fa-upload me-2"></i> Subir y Reentrenar Modelo';
    }
}

// ===== FUNCIONES DE PROGRESO =====
function updateProgress(percent, stepText) {
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const currentStep = document.getElementById('currentStep');
    
    if (progressFill) progressFill.style.width = percent + '%';
    if (progressText) progressText.textContent = percent + '%';
    if (currentStep) currentStep.textContent = stepText;
}

function updateStatusBadge(type, message) {
    const statusBadge = document.getElementById('statusBadge');
    if (!statusBadge) return;
    
    const icons = {
        success: 'fas fa-check-circle',
        error: 'fas fa-exclamation-triangle',
        processing: 'fas fa-cogs'
    };
    
    statusBadge.innerHTML = `<i class="${icons[type] || icons.processing}"></i> ${message}`;
    statusBadge.className = `status-badge status-${type}`;
}

// ===== ANIMACIONES Y EFECTOS VISUALES =====
function initializeAnimations() {
    // Animaciones para campos de formulario
    document.querySelectorAll('.form-control, .form-select').forEach(element => {
        element.addEventListener('focus', function() {
            this.parentElement.style.transform = 'scale(1.02)';
            this.parentElement.style.transition = 'transform 0.3s ease';
        });
        
        element.addEventListener('blur', function() {
            this.parentElement.style.transform = 'scale(1)';
        });
    });

    // Animaciones para elementos de recomendación
    document.querySelectorAll('.recommendation-item').forEach(item => {
        item.addEventListener('click', function() {
            this.style.transform = 'scale(0.98)';
            setTimeout(() => {
                this.style.transform = 'translateX(5px)';
            }, 100);
        });
    });

    // Efecto parallax para el header
    window.addEventListener('scroll', () => {
        const header = document.querySelector('.header-section');
        if (header) {
            const scrolled = window.pageYOffset;
            const rate = scrolled * -0.5;
            header.style.transform = `translateY(${rate}px)`;
        }
    });
}

function showLoadingAnimation() {
    const loadingSpinner = document.getElementById('loadingSpinner');
    if (loadingSpinner) {
        loadingSpinner.style.display = 'block';
        
        // Ocultar después de un tiempo
        setTimeout(() => {
            loadingSpinner.style.display = 'none';
        }, 3000);
    }
}

function playSuccessAnimation(element) {
    if (!element) return;
    
    element.classList.add('success-animation');
    setTimeout(() => {
        element.classList.remove('success-animation');
    }, 600);
}

function highlightField(field, color) {
    if (!field) return;
    
    const originalColor = field.style.backgroundColor;
    field.style.backgroundColor = color;
    field.style.transition = 'background-color 0.3s ease';
    
    setTimeout(() => {
        field.style.backgroundColor = originalColor;
    }, 1500);
}

// ===== VALIDACIONES =====
function initializeValidations() {
    // Validación en tiempo real para edad
    const edadInput = document.getElementById('edad');
    if (edadInput) {
        edadInput.addEventListener('input', function() {
            const edad = parseInt(this.value);
            const feedback = this.parentElement.querySelector('.invalid-feedback') || 
                           createFeedbackElement(this.parentElement);
            
            if (edad < 5 || edad > 25) {
                this.classList.add('is-invalid');
                feedback.textContent = 'La edad debe estar entre 5 y 25 años';
            } else {
                this.classList.remove('is-invalid');
                this.classList.add('is-valid');
                feedback.textContent = '';
            }
        });
    }

    // Validación para grado
    const gradoInput = document.getElementById('grado');
    if (gradoInput) {
        gradoInput.addEventListener('input', function() {
            const grado = parseInt(this.value);
            const feedback = this.parentElement.querySelector('.invalid-feedback') || 
                           createFeedbackElement(this.parentElement);
            
            if (grado < 1 || grado > 12) {
                this.classList.add('is-invalid');
                feedback.textContent = 'El grado debe estar entre 1 y 12';
            } else {
                this.classList.remove('is-invalid');
                this.classList.add('is-valid');
                feedback.textContent = '';
            }
        });
    }

    // Validación para tiempo
    const tiempoInput = document.getElementById('tiempo_seg');
    if (tiempoInput) {
        tiempoInput.addEventListener('input', function() {
            const tiempo = parseInt(this.value);
            const feedback = this.parentElement.querySelector('.invalid-feedback') || 
                           createFeedbackElement(this.parentElement);
            
            if (tiempo < 1) {
                this.classList.add('is-invalid');
                feedback.textContent = 'El tiempo debe ser mayor a 0 segundos';
            } else if (tiempo > 3600) {
                this.classList.add('is-invalid');
                feedback.textContent = 'El tiempo parece muy alto (máximo 1 hora)';
            } else {
                this.classList.remove('is-invalid');
                this.classList.add('is-valid');
                feedback.textContent = '';
            }
        });
    }
}

function createFeedbackElement(parent) {
    const feedback = document.createElement('div');
    feedback.className = 'invalid-feedback';
    parent.appendChild(feedback);
    return feedback;
}

// ===== UTILIDADES =====
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function showAlert(type, message, duration = 5000) {
    // Crear elemento de alerta
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertDiv.style.cssText = `
        top: 20px;
        right: 20px;
        z-index: 9999;
        min-width: 300px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border-radius: 10px;
    `;
    
    const icons = {
        success: 'fas fa-check-circle',
        error: 'fas fa-exclamation-triangle',
        warning: 'fas fa-exclamation-circle',
        info: 'fas fa-info-circle'
    };
    
    alertDiv.innerHTML = `
        <i class="${icons[type] || icons.info} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // Auto-remover después del tiempo especificado
    setTimeout(() => {
        if (alertDiv && alertDiv.parentNode) {
            alertDiv.classList.remove('show');
            setTimeout(() => alertDiv.remove(), 150);
        }
    }, duration);
    
    return alertDiv;
}

function showTooltip(element, message, duration = 3000) {
    if (!element) return;
    
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip-custom';
    tooltip.textContent = message;
    tooltip.style.cssText = `
        position: absolute;
        background: rgba(0,0,0,0.8);
        color: white;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.875rem;
        z-index: 1000;
        pointer-events: none;
        transform: translateX(-50%);
        white-space: nowrap;
    `;
    
    // Posicionar tooltip
    const rect = element.getBoundingClientRect();
    tooltip.style.left = (rect.left + rect.width / 2) + 'px';
    tooltip.style.top = (rect.top - 35) + 'px';
    
    document.body.appendChild(tooltip);
    
    // Animar entrada
    tooltip.style.opacity = '0';
    tooltip.style.transform += ' translateY(10px)';
    tooltip.style.transition = 'all 0.3s ease';
    
    requestAnimationFrame(() => {
        tooltip.style.opacity = '1';
        tooltip.style.transform = tooltip.style.transform.replace('translateY(10px)', 'translateY(0)');
    });
    
    // Remover después del tiempo especificado
    setTimeout(() => {
        if (tooltip && tooltip.parentNode) {
            tooltip.style.opacity = '0';
            tooltip.style.transform += ' translateY(-10px)';
            setTimeout(() => tooltip.remove(), 300);
        }
    }, duration);
}

function showWelcomeMessage() {
    // Mostrar mensaje de bienvenida solo la primera vez
    if (!localStorage.getItem('welcomeShown')) {
        setTimeout(() => {
            showAlert('info', '¡Bienvenido al Sistema de Predicción Estudiantil! Completa el formulario para obtener predicciones personalizadas.', 7000);
            localStorage.setItem('welcomeShown', 'true');
        }, 1000);
    }
}

// ===== FUNCIONES DE API =====
async function checkModelStatus() {
    try {
        const response = await fetch('/api/model_info');
        const data = await response.json();
        return data.modelo_activo || false;
    } catch (error) {
        console.error('Error verificando estado del modelo:', error);
        return false;
    }
}

async function getSystemHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error verificando salud del sistema:', error);
        return { status: 'error', error: error.message };
    }
}

// ===== MANEJO DE ERRORES GLOBALES =====
window.addEventListener('error', function(e) {
    console.error('Error global:', e.error);
    showAlert('error', 'Se produjo un error inesperado. Por favor, recarga la página.');
});

window.addEventListener('unhandledrejection', function(e) {
    console.error('Promesa rechazada:', e.reason);
    showAlert('error', 'Error de conexión. Verifica tu conexión a internet.');
});

// ===== FUNCIÓN DE ACTUALIZACIÓN DE ESTADO =====
function updateSystemStatus() {
    getSystemHealth().then(health => {
        const statusIndicator = document.getElementById('systemStatus');
        if (statusIndicator) {
            const isHealthy = health.status === 'healthy';
            statusIndicator.className = `badge ${isHealthy ? 'bg-success' : 'bg-danger'}`;
            statusIndicator.textContent = isHealthy ? 'Sistema Activo' : 'Sistema con Errores';
        }
    });
}

// ===== INICIALIZAR ACTUALIZACIONES PERIÓDICAS =====
setInterval(updateSystemStatus, 30000); // Cada 30 segundos

// ===== EXPORT DE FUNCIONES PRINCIPALES =====
// Para uso desde HTML inline si es necesario
window.StudentPredictionSystem = {
    uploadCSV,
    showAlert,
    showTooltip,
    updateProgress,
    formatFileSize,
    checkModelStatus,
    getSystemHealth
};