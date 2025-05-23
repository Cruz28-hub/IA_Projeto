import glob

# Caminho para as pastas de labels
pastas = [
    r'C:\Users\luiss\Documents\Estagio\IA_Projeto\super_tux_kart_project_luis-4\train\labels',
    r'C:\Users\luiss\Documents\Estagio\IA_Projeto\super_tux_kart_project_luis-4\valid\labels',
    r'C:\Users\luiss\Documents\Estagio\IA_Projeto\super_tux_kart_project_luis-4\test\labels'
]

nc = 8  # número de classes

for pasta in pastas:
    for file in glob.glob(f"{pasta}\\*.txt"):
        with open(file, "r") as f:
            for i, line in enumerate(f):
                if line.strip() == "":
                    continue
                class_idx = int(line.strip().split()[0])
                if class_idx >= nc or class_idx < 0:
                    print(f"Arquivo {file}, linha {i+1}: índice de classe inválido {class_idx}")
print("Verificação concluída. Nenhuma classe inválida encontrada.")