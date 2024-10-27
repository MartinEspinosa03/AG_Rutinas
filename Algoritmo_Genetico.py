import json
import os
import random
import csv
import matplotlib.pyplot as plt

class AlgoritmoGenetico:
    def __init__(self, tipo_rutina, selected_categories=None):
        self.tipo_rutina = tipo_rutina
        self.selected_categories = selected_categories if selected_categories else []
        print("Importando dataset...")

        if self.tipo_rutina == "ganar_fuerza":
            data = self.cargar_json("json/RutinasGanarFuerza.json")
        elif self.tipo_rutina == "ganar_masa":
            data = self.cargar_json("json/RutinasGanarMasa.json")
        elif self.tipo_rutina == "resistencia":
            data = self.cargar_json("json/RutinasResistencia.json")
        else:
            raise ValueError("Tipo de rutina no válido.")

        self.data = []
        for category in data:
            self.data.extend(data[category])

        for item in self.data:
            print(f"Elemento en data: {item}, tipo: {type(item)}")

        if self.tipo_rutina == "ganar_masa" and self.selected_categories:
            self.filtrar_por_categorias()

        self.imprimir_ejercicios_disponibles()
        self.verificar_integridad_datos()

        self.bit_size = self.calculate_bit_size(len(self.data))
        self.readed_ids = []
        self.readed_categories = self.cargar_json("json/categorias.json")
        self.readed_exercises = self.cargar_json("json/Ejercicios.json")
        self.favorites_exercises, self.favorites_categories = self.definir_favoritos()

        self.best_individual = {"gen": None, "weight": None}
        self.worst_individual = {"gen": None, "weight": None}

    def cargar_json(self, path):
        with open(path, "r", encoding="utf8") as f:
            return json.load(f)

    def filtrar_por_categorias(self):
        self.data = [exercise for exercise in self.data if isinstance(exercise, dict) and any(cat in exercise.get('categorias', []) for cat in self.selected_categories)]

    def imprimir_ejercicios_disponibles(self):
        print(f"Ejercicios disponibles: {[exercise.get('nombre', 'Desconocido') for exercise in self.data]}")

    def verificar_integridad_datos(self):
        
        for index, item in enumerate(self.data):
            if 'id' not in item:
                raise KeyError(f"El elemento en la posición {index} no tiene la clave 'id'")

    def definir_favoritos(self):
        favorites_exercises = set()
        favorites_categories = set()

        if self.tipo_rutina == "ganar_fuerza":
            favorites_exercises.update(["Paused Bench Press", "Squat", "Deadlift"])
            favorites_categories.update(["PECHO", "PIERNA"])
        elif self.tipo_rutina == "ganar_masa":
            favorites_categories.update(self.selected_categories)
        elif self.tipo_rutina == "resistencia":
            favorites_exercises.update(["Burpees", "Mountain Climbers", "Kettlebell Swings"])
            favorites_categories.update(["Circuito 1", "Circuito 2", "Circuito 3"])

        return favorites_exercises, favorites_categories

    def calculate_bit_size(self, numero):
        return len(bin(numero)) - 2

    def setup(self, porcentaje_cruza=0.5, porcentaje_mut_ind=0.5, porcentaje_mut_gen=0.5, poblacion_inicial=10, poblacion_max=40, generaciones=100):
        self.cross_prob = float(porcentaje_cruza) 
        self.individual_mut_prob = float(porcentaje_mut_ind) 
        self.gen_mut_prob = float(porcentaje_mut_gen)  
        self.initial_population = int(poblacion_inicial) 
        self.max_population = int(poblacion_max)  
        self.generations_count = int(generaciones) 

    def random_individual_gen(self):
        categories = self.selected_categories if self.tipo_rutina == "ganar_masa" else \
                    ["HOMBRO", "PECHO", "ESPALDA", "PIERNA", "BICEPS", "TRICEPS"]

        selected_exercises = []
        for category in categories:
            exercises_in_category = [idx for idx, exercise in enumerate(self.data) if category in exercise.get('categorias', [])]
            if exercises_in_category:
                num_exercises_to_select = min(4, len(exercises_in_category))  
                selected_exercises.extend(random.sample(exercises_in_category, num_exercises_to_select))

        return tuple(selected_exercises) if selected_exercises else tuple()


    def invoke_first_generation(self):
        return [self.random_individual_gen() for _ in range(self.initial_population)]

    def poda(self, population):
        while len(population) > self.max_population:
            index_to_delete = random.randrange(0, len(population))
            if population[index_to_delete] == self.best_individual:
                continue 
            del population[index_to_delete]
        return population

    def calculate_from_population(self, gen_list):
        logs = []
        best = {'gen': None, 'weight': float('-inf')}
        worst = {'gen': None, 'weight': float('inf')}
        total_weight = 0

        for individual in gen_list:
            individual_weight = self.evaluate_individual(individual)
            logs.append({'gen': individual, 'weight': individual_weight})
            total_weight += individual_weight

            if individual_weight > best['weight']:
                best = {'gen': individual, 'weight': individual_weight}
            if individual_weight < worst['weight']:
                worst = {'gen': individual, 'weight': individual_weight}

        average_weight = round(total_weight / len(gen_list), 3) if gen_list else 0
        return {'logs': logs, 'stats': {'best': best, 'worst': worst, 'average': average_weight}}

    def cross(self, population):
        result = population[:]
        max_partners = max(1, int(len(population) * 0.2))

        for index, individual in enumerate(population):
            num_crosses = random.randint(1, max_partners)
            
            for _ in range(num_crosses):
                if random.uniform(0.01, 1.00) <= self.cross_prob:
                    individual_to_cross = random.choice([x for i, x in enumerate(population) if i != index])
                    crossed_individuals = self.exchange_gen(individual, individual_to_cross)
                    result.extend(crossed_individual for crossed_individual in crossed_individuals if len(set(crossed_individual)) == len(crossed_individual))

            if random.uniform(0.01, 1.00) <= self.individual_mut_prob:
                mutated_individual = self.mutate_individual(individual)
                if len(set(mutated_individual)) == len(mutated_individual):
                    result.append(mutated_individual)

        return result

    def mutate_individual(self, individual):
        mutated_individual = list(individual)
        indices_ejercicios = [idx for idx in range(len(self.data)) if self.data[idx]['id'] not in self.readed_ids]

        for i in range(len(mutated_individual)):
            if random.uniform(0.01, 1.00) <= self.gen_mut_prob:
                posibles_nuevos_ejercicios = [idx for idx in indices_ejercicios if idx not in mutated_individual and self.data[idx]['id'] not in self.readed_ids]
                if posibles_nuevos_ejercicios:
                    nuevo_ejercicio = random.choice(posibles_nuevos_ejercicios)
                    mutated_individual[i] = nuevo_ejercicio

        if random.uniform(0.01, 1.00) <= self.gen_mut_prob:
            possible_ab_exercises = [idx for idx, exercise in enumerate(self.data) if "ABDOMEN" in exercise.get('categorias', [])]
            if possible_ab_exercises:
                ab_exercise = random.choice(possible_ab_exercises)
                mutated_individual.append(ab_exercise)

        return tuple(mutated_individual)

    def exchange_gen(self, original_individual, individual_to_cross):
        if len(original_individual) < 2 or len(individual_to_cross) < 2:
            return [original_individual, individual_to_cross]

        points_of_crossover = sorted(random.sample(range(1, len(original_individual)), random.randint(1, len(original_individual) - 1)))
        new_individual1, new_individual2 = list(original_individual), list(individual_to_cross)
        for i, point in enumerate(points_of_crossover):
            if i % 2 == 1: 
                new_individual1[point:], new_individual2[point:] = new_individual2[point:], new_individual1[point:]
        return [tuple(new_individual1), tuple(new_individual2)]

    def search_from_index(self, exercises_list):
        result = []

        for exercise_index_bin in exercises_list:
            individual_as_index = exercise_index_bin["gen"]

            if individual_as_index >= len(self.data):
                print(f"Saltando ejercicio {exercise_index_bin}, fuera de rango.")
                continue

            exercise_data = self.data[individual_as_index]
            exercise_data["weight"] = exercise_index_bin["weight"]
            exercise_data["matching_details"] = exercise_data["descripcion"]
            result.append(exercise_data)

        return result

    def render_graphics(self):
        plt.clf()
        
        average_values = [entry['average'] for entry in self.all_records]
        best_weights = [entry['best']['weight'] for entry in self.all_records]
        worst_weights = [entry['worst']['weight'] for entry in self.all_records]

        indices = list(range(len(self.all_records)))

        plt.figure(figsize=(12, 6))
        plt.plot(indices, average_values, marker='o', linestyle='-', label='Aptitud promedio')
        plt.plot(indices, best_weights, marker='o', linestyle='-', label='Mejor Aptitud')
        plt.plot(indices, worst_weights, marker='o', linestyle='-', label='Peor Aptitud')

        plt.xlabel('Generación')
        plt.ylabel('Aptitud')
        plt.title('Reporte Histórico')
        plt.legend()

        plt.grid(True)
        carpeta = "graphics"
        if not os.path.exists(carpeta):
            os.makedirs(carpeta)

        ruta_grafica = os.path.join(carpeta, "grafica.png")
        plt.savefig(ruta_grafica)
        plt.close()
        print(f"Gráfica guardada en {ruta_grafica}")
        return ruta_grafica

    def evaluate_individual(self, individual):
        individual_weight = 0
        print(f"Evaluando individuo: {individual}")
        
        for exercise_index in individual:
            if exercise_index < len(self.data):
                exercise_data = self.data[exercise_index]
                exercise_weight = 0

                if 'ejercicios' in exercise_data:
                    for exercise in exercise_data['ejercicios']:
                        if exercise in self.readed_exercises:
                            exercise_weight -= 1  
                        if exercise in self.favorites_exercises:
                            exercise_weight += 2  

                if 'categorias' in exercise_data:
                    for category in exercise_data['categorias']:
                        if category in self.readed_categories:
                            exercise_weight -= 1  
                        if category in self.favorites_categories:
                            exercise_weight += 2 

                individual_weight += exercise_weight
            else:
                print(f"Índice de ejercicio fuera de rango: {exercise_index}")

        print(f"Peso del individuo: {individual_weight}")
        return individual_weight

    def save_results_to_csv(self, results):
        csv_file_path = os.path.join('C:\\8Cuatri\\IA\\AG_Rutinas\\graphics', 'resultados_finales.csv')
        
        if not os.path.exists(os.path.dirname(csv_file_path)):
            os.makedirs(os.path.dirname(csv_file_path))
        
        with open(csv_file_path, 'w', newline='', encoding='utf8') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Título', 'Ejercicios', 'Categorías', 'Descripción', "Series", "Repeticiones"])
            for result in results:
                writer.writerow([result['id'], result['ejercicio'], result.get('ejercicios', ''), result.get('categorias', ''), result.get('descripcion', ''), result.get('series', ''), result.get('repeticiones', '')])

    def run(self):
        self.best_individual = {"gen": None, "weight": None}
        self.worst_individual = {"gen": None, "weight": None}
        last_average_weight = None
        no_change_generations = 0

        self.all_records = []
        population = self.invoke_first_generation()
        print("Primer generación:", population)

        for i in range(self.generations_count):
            print(f"Generación {i+1} de {self.generations_count}")
            population = self.cross(population)

            if self.best_individual["gen"] is not None:
                if self.best_individual["gen"] not in population:
                    population[random.randrange(0, len(population))] = self.best_individual["gen"]

            result = self.calculate_from_population(population)

            if self.best_individual["gen"] is not None:
                self.best_individual = self.best_individual if self.best_individual["weight"] > result["stats"]["best"]["weight"] else result["stats"]["best"]
            else:
                self.best_individual = result["stats"]["best"]

            if self.worst_individual["gen"] is not None:
                self.worst_individual = self.worst_individual if self.worst_individual["weight"] < result["stats"]["worst"]["weight"] else result["stats"]["worst"]
            else:
                self.worst_individual = result["stats"]["worst"]

            result["stats"]["best"] = self.best_individual
            self.all_records.append(result["stats"])

            current_average_weight = result["stats"]["average"]
            if last_average_weight is not None and current_average_weight == last_average_weight:
                no_change_generations += 1
            else:
                no_change_generations = 0

            last_average_weight = current_average_weight

            if no_change_generations >= 10:
                print(f"Convergencia alcanzada después de {i+1} generaciones.")
                break

            if len(population) > self.max_population:
                population = self.poda(population)

        best_exercises = self.search_from_index([{'gen': idx, 'weight': 0} for idx in self.best_individual["gen"]])
        self.save_results_to_csv(best_exercises)
        self.render_graphics()
        return best_exercises, self.best_individual['weight']

def main():
    tipo_rutina = "resistencia"
    ag = AlgoritmoGenetico(tipo_rutina)
    ag.setup(porcentaje_cruza=0.5, porcentaje_mut_ind=0.5, porcentaje_mut_gen=0.5, poblacion_inicial=10, poblacion_max=40, generaciones=100)
    best_exercises, best_weight = ag.run()
    print(f"Mejor individuo encontrado: {best_exercises} con un peso de {best_weight}")

if __name__ == "__main__":
    main()
