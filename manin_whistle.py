from manim import *
from numpy import *


class VerticeWhistle(Dot):
    """
    Класс, отвечающий за математическую реализацию точек
    """
    def __init__(self, coordinates: ndarray, **kwargs):
        """

        :param coordinates: начальные координаты точки на сцене
        :param kwargs: остальные параметры для Dot
        """
        self.kwargs = kwargs

        super().__init__(**kwargs)
        self.move_to(coordinates)

    def get_perpendicular_to(self, vector: 'VectorWhistle', **kwargs) -> 'VectorWhistle':
        """
        Метод, возвращающий перпендикуляр к другому вектору
        :param vector: вектор, к которому требуется провести перпендикуляр
        :return: перпендикуляр к vector с основанием в self
        """
        unit_normal = vector.get_unit_normal(start_vertice=self)
        H_vertice = unit_normal.get_intersection_with(vector)
        return VectorWhistle(self, H_vertice, **kwargs)

    def get_median_to(self, vector: 'VectorWhistle', **kwargs) -> 'VectorWhistle':
        """
        Метод, возвращающий вектор с началом в self и концом в середине vector
        :param vector: вектор, к центру которого проводится искомый вектор
        :return: вектор с началом в self и концом в середине vector
        """
        x = (vector.vertice_1.x + vector.vertice_2.x) / 2
        y = (vector.vertice_1.y + vector.vertice_2.y) / 2
        return VectorWhistle(self, VerticeWhistle(np.array([x, y, 0])), **kwargs)

    @property
    def coordinates(self) -> ndarray:
        """
        Свойство coordinates отвечает за координаты точки
        Реализован свойством, потому что обновляется в реальном времени, даже во время анимации
        :return: ndarray[x, y, 0] - центр точки
        """
        return self.get_center()

    @property
    def x(self) -> float:
        """
        Свойство х отвечает за координату х точки
        Реализован свойством, потому что обновляется в реальном времени, даже во время анимации
        :return: coordinates[0]
        """
        return self.get_center()[0]

    @property
    def y(self) -> float:
        """
        Свойство y отвечает за координату y точки
        Реализован свойством, потому что обновляется в реальном времени, даже во время анимации
        :return: coordinates[1]
        """
        return self.get_center()[1]

    def get_always_redraw(self) -> 'VerticeWhistle':
        """
        Метод, возвращающий объект VerticeWhistle, который будет перерисовываться каждый кадр
        :return: VerticeWhistle (Mobject - дальний родитель VerticeWhistle)
        """
        return always_redraw(lambda: VerticeWhistle(self.coordinates, **self.kwargs))

    def set_params(self, **kwargs):
        return VerticeWhistle(coordinates=self.coordinates, **kwargs)


class VectorWhistle(Line):
    """
    Класс, отвечающий за математическую реализацию векторов
    """
    def __init__(self, vertice_1: VerticeWhistle, vertice_2: VerticeWhistle, **kwargs):
        """

        :param vertice_1: Начальная вершина вектора
        :param vertice_2: Конечная вершина вектора
        :param kwargs: Параметры для Line
        """
        super().__init__(vertice_1.coordinates, vertice_2.coordinates, **kwargs)
        self.kwargs = kwargs

        self.vertice_1 = vertice_1
        self.vertice_2 = vertice_2

    def scalar_product_with(self, vector: 'VectorWhistle') -> float:
        """
        Метод, возвращающий скалярное произведение двух векторов
        :param vector: вектор, который нужно скалярно умножить на self
        :return: скалярное произведение
        """
        return self.coordinates[0] * vector.coordinates[0] + self.coordinates[1] * vector.coordinates[1]

    def get_reversed(self) -> 'VectorWhistle':
        """
        Метод, возвращающий вектор, равный по модулю и противоположный по направлению вектору self
        :return: Противоположный вектор
        """
        return VectorWhistle(self.vertice_2, self.vertice_1, **self.kwargs)

    def get_intersection_with(self, vector: 'VectorWhistle', **kwargs) -> VerticeWhistle:
        """
        Метод, возвращающий точку пересечения двух векторов
        Если вектора направляющие, то возвращается точка пересечения прямых, на которых соответственно лежат вектора
        :param vector: вектор, точку пересечения с которым необходимо вычислить
        :return: точка пересечения векторов (VerticeWhistle)
        """

        # Реализовано через систему уравнений двух коллинеарных векторов ↓
        x = ((vector.vertice_1.y - self.vertice_1.y) * self.x * vector.x + self.y * vector.x * self.vertice_1.x - self.x * vector.y * vector.vertice_1.x) / (self.y * vector.x - self.x * vector.y)
        y = (x - self.vertice_1.x) / self.x * self.y + self.vertice_1.y
        return VerticeWhistle(coordinates=np.array([x, y, 0]), **kwargs)

    def get_line_center(self, **kwargs):
        return VerticeWhistle(
            np.array([(self.vertice_1.x + self.vertice_2.x) / 2, (self.vertice_1.y + self.vertice_2.y) / 2, 0]), **kwargs)

    def get_unit_normal(self, start_vertice: VerticeWhistle, **kwargs):
        """
        Метод, возвращающий направляющий вектор, перпендикулярный вектору self
        :param start_vertice: начальная вершина вектора
        :return: вектор, перпендикулярный вектору self с определенной начальной точкой
        """
        # Реализовано через уравнение скалярного произведения ↓
        return VectorWhistle(start_vertice, VerticeWhistle(
            np.array([start_vertice.x + 1, start_vertice.y - self.coordinates[0] / self.coordinates[1], 0])), **kwargs)

    def get_always_redraw(self) -> 'VectorWhistle':
        """
        Метод, возвращающий объект VectorWhistle, который будет перерисовываться каждый кадр
        :return: VectorWhistle (Mobject - дальний родитель VectorWhistle)
        """
        return always_redraw(lambda: VectorWhistle(self.vertice_1, self.vertice_2, **self.kwargs))

    def set_params(self, **kwargs):
        return VectorWhistle(self.vertice_1, self.vertice_2, **kwargs)

    @property
    def coordinates(self) -> ndarray:
        """
        Свойство coordinates отвечает за координаты радиус-вектора self
        Реализован свойством, потому что обновляется в реальном времени, даже во время анимации
        :return: ndarray[x, y, 0] - координаты радиус-вектора self
        """
        return np.array([self.vertice_2.x - self.vertice_1.x, self.vertice_2.y - self.vertice_1.y, 0])

    @property
    def x(self) -> float:
        """
        Свойство х отвечает за координату х радиус-вектора self
        Реализован свойством, потому что обновляется в реальном времени, даже во время анимации
        :return: coordinates[0]
        """
        return self.coordinates[0]

    @property
    def y(self) -> float:
        """
        Свойство y отвечает за координату y радиус-вектора self
        Реализован свойством, потому что обновляется в реальном времени, даже во время анимации
        :return: coordinates[1]
        """
        return self.coordinates[1]

    @property
    def length(self) -> float:
        """
        Свойство length отвечает за длину вектора self
        Реализован свойством, потому что обновляется в реальном времени, даже во время анимации
        :return: длина вектора self
        """
        return sqrt(self.coordinates[0] ** 2 + self.coordinates[1] ** 2)


class AngleWhistle(Angle):
    """
    Метод, отвечающий за математическую реализацию углов
    """
    def __init__(self, vector_1: VectorWhistle, vector_2: VectorWhistle, **kwargs):
        """

        !!!(vector_1.vertice_1 = vector_2.vertice_1)!!! - вектора непременно должны иметь общее начало
        :param vector_1: вектор, образующий угол вместе с vector_2
        :param vector_2: вектор, образующий угол вместе с vector_1
        :param kwargs: Параметры для Angle
        """
        super().__init__(vector_1, vector_2, **kwargs)
        self.kwargs = kwargs

        self.vector_1 = vector_1
        self.vector_2 = vector_2

    @property
    def rad(self):
        """
        Свойство length отвечает за радиальную меру угла self
        :return: угол self в радианах
        """
        # Реализовано через уравнение скалярного произведения ↓
        return arccos(self.vector_1.scalar_product_with(self.vector_2) / (self.vector_1.length * self.vector_2.length))

    def get_unit_bisector(self, **kwargs) -> VectorWhistle:
        """
        Метод, возвращающий направляющий вектор биссектрисы угла self
        :return: направляющий вектор биссектрисы VectorWhistle
        """
        # Реализовано через уравнение скалярного произведения ↓
        alpha = (self.vector_1.length * self.vector_2.x - self.vector_2.length * self.vector_1.x) / (
                    self.vector_2.length * self.vector_1.y - self.vector_1.length * self.vector_2.y)
        return VectorWhistle(self.vector_1.vertice_1, VerticeWhistle(
            np.array([self.vector_1.vertice_1.x + 1, self.vector_1.vertice_1.y + alpha, 0])), **kwargs)

    def get_bisector_to(self, vector: VectorWhistle, **kwargs):
        """
        Метод, возвращающий вектор, содержащий биссектрису угла self и ограниченный другим вектором vector
        :param vector: вектор, который определяет длину биссектрисы (в треугольнике - противоположная сторона)
        :return: вектор, содержащий биссектрису угла self
        """
        # ab - 1, ac - 2
        return VectorWhistle(self.vector_1.vertice_1, vector.get_intersection_with(self.get_unit_bisector()), **kwargs)

    def get_adjacent(self, ground_vector: VectorWhistle, **kwargs):
        """
        Метод, возвращающий угол, смежный с углом self
        :param ground_vector: вектор, содержащий сторону, которую нужно продлить
        :return: угол, смежный с self
        """
        if ground_vector == self.vector_1:
            new_vec = VectorWhistle(self.vector_1.vertice_1, VerticeWhistle(coordinates=np.array([self.vector_1.vertice_1.x + self.vector_1.get_reversed().x, self.vector_1.vertice_1.y + self.vector_1.get_reversed().y, 0])))
            return AngleWhistle(new_vec, self.vector_2, **kwargs)
        elif ground_vector == self.vector_2:
            new_vec = VectorWhistle(self.vector_2.vertice_1, VerticeWhistle(coordinates=np.array(
                [self.vector_2.vertice_1.x + self.vector_2.get_reversed().x,
                 self.vector_2.vertice_1.y + self.vector_2.get_reversed().y, 0])))
            return AngleWhistle(new_vec, self.vector_1, **kwargs)

    def get_always_redraw(self) -> 'AngleWhistle':
        """
        Метод, возвращающий объект VectorWhistle, который будет перерисовываться каждый кадр
        :return: VectorWhistle (Mobject - дальний родитель VectorWhistle)
        """
        return always_redraw(lambda: AngleWhistle(self.vector_1, self.vector_2, **self.kwargs))

    def set_params(self, **kwargs):
        return AngleWhistle(self.vector_1, self.vector_2, **kwargs)

    def make_right(self, **kwargs):
        return RightAngle(line1=self.vector_1, line2=self.vector_2, **kwargs)


class TriangleWhistle(Polygon):
    """
    Класс, отвечающий за математическую реализацию треугольника
    """
    def __init__(self, A_vertice: VerticeWhistle, B_vertice: VerticeWhistle, C_vertice: VerticeWhistle, always_redraw_bool: bool, **kwargs):
        """

        :param A_vertice: вершина A треугольника
        :param B_vertice: вершина B треугольника
        :param C_vertice: вершина C треугольника
        :param kwargs: дополнительные параметры для Polygon
        """
        super().__init__(A_vertice.coordinates, B_vertice.coordinates, C_vertice.coordinates, **kwargs)
        if always_redraw_bool:
            self.A_vertice = A_vertice.get_always_redraw()
            self.B_vertice = B_vertice.get_always_redraw()
            self.C_vertice = C_vertice.get_always_redraw()

            self.AB_vector = VectorWhistle(self.A_vertice, self.B_vertice).get_always_redraw()
            self.BC_vector = VectorWhistle(self.B_vertice, self.C_vertice).get_always_redraw()
            self.CA_vector = VectorWhistle(self.C_vertice, self.A_vertice).get_always_redraw()
            self.AC_vector = self.CA_vector.get_reversed().get_always_redraw()
            self.BA_vector = self.AB_vector.get_reversed().get_always_redraw()
            self.CB_vector = self.BC_vector.get_reversed().get_always_redraw()

            self.alpha_angle = AngleWhistle(self.AB_vector, self.AC_vector).get_always_redraw()
            self.beta_angle = AngleWhistle(self.BC_vector, self.BA_vector).get_always_redraw()
            self.gamma_angle = AngleWhistle(self.CA_vector, self.CB_vector).get_always_redraw()
        else:
            self.A_vertice = A_vertice
            self.B_vertice = B_vertice
            self.C_vertice = C_vertice

            self.AB_vector = VectorWhistle(self.A_vertice, self.B_vertice)
            self.BC_vector = VectorWhistle(self.B_vertice, self.C_vertice)
            self.CA_vector = VectorWhistle(self.C_vertice, self.A_vertice)
            self.AC_vector = self.CA_vector.get_reversed()
            self.BA_vector = self.AB_vector.get_reversed()
            self.CB_vector = self.BC_vector.get_reversed()

            self.alpha_angle = AngleWhistle(self.AB_vector, self.AC_vector)
            self.beta_angle = AngleWhistle(self.BC_vector, self.BA_vector)
            self.gamma_angle = AngleWhistle(self.CA_vector, self.CB_vector)

    def get_median_to_side(self, vertice: VerticeWhistle, **kwargs) -> VectorWhistle:
        """
        Метод, возвращающий вектор, содержащий медиану
        :param vertice: вершина медианы
        :param kwargs: параметры для Line
        :return: медиана
        """
        if vertice.x == self.A_vertice.x:
            # BC
            return vertice.get_median_to(self.BC_vector, **kwargs)
        elif vertice.x == self.B_vertice.x:
            # AC
            return vertice.get_median_to(self.CA_vector, **kwargs)
        elif vertice.x == self.C_vertice.x:
            # AB
            return vertice.get_median_to(self.AB_vector, **kwargs)
        else:
            # nothing
            raise ValueError

    def get_height_to_side(self, vertice: VerticeWhistle, **kwargs):
        """
        Метод, возвращающий вектор, содержащий высоту
        :param vertice: вершина высоты
        :param kwargs: параметры для Line
        :return: высота
        """
        if vertice.x == self.A_vertice.x:
            # BC
            return vertice.get_perpendicular_to(self.BC_vector, **kwargs)
        elif vertice.x == self.B_vertice.x:
            # AC
            return vertice.get_perpendicular_to(self.CA_vector.get_reversed(), **kwargs)
        elif vertice.x == self.C_vertice.x:
            # AB
            return vertice.get_perpendicular_to(self.AB_vector, **kwargs)
        else:
            # nothing
            raise ValueError

    def get_bisector_to_side(self, vertice: VerticeWhistle, **kwargs):
        """
        Метод, возвращающий вектор, содержащий биссектрису
        :param vertice: вершина угла
        :param kwargs: параметры для Line
        :return: биссектриса
        """
        if vertice.x == self.A_vertice.x:
            # BC
            return self.alpha_angle.get_bisector_to(self.BC_vector, **kwargs)
        elif vertice.x == self.B_vertice.x:
            # AC
            return self.beta_angle.get_bisector_to(self.CA_vector.get_reversed(), **kwargs)
        elif vertice.x == self.C_vertice.x:
            # AB
            return self.gamma_angle.get_bisector_to(self.AB_vector, **kwargs)
        else:
            # nothing
            raise ValueError

    def get_incenter(self, **kwargs):
        """
        Метод, возвращающий вершину центра вписанной окружности
        :param kwargs: параметры для Dot
        :return: центр вписанной окружности
        """
        return self.get_bisector_to_side(vertice=self.A_vertice).get_intersection_with(
            self.get_bisector_to_side(vertice=self.B_vertice), **kwargs)

    def get_circumscribed_center(self, **kwargs):
        """
        Метод, возвращающий вершину центра описанной окружности
        :return: центр описанной окружности
        """
        return self.AB_vector.get_unit_normal(start_vertice=self.AB_vector.get_line_center()).get_intersection_with(
            self.BC_vector.get_unit_normal(start_vertice=self.BC_vector.get_line_center()), **kwargs)

    def get_incenter_radius(self):
        """
        Метод, возвращающий радиус вписанной окружности
        :return: радиус вписанной окружности
        """
        return (self.AB_vector.length * self.CA_vector.length * sin(self.alpha_angle.rad)) / (
                    self.AB_vector.length + self.BC_vector.length + self.CA_vector.length)

    def get_circumscribed_radius(self):
        """
        Метод, возвращающий радиус описанной окружности
        :return: радиус описанной окружности
        """
        return self.BC_vector.length / (2 * sin(self.alpha_angle.rad))

    def get_enscribed_radius(self, touching_side: VectorWhistle):
        """
        Метод, возвращающий радиус вневписанной окружности, касающийся стороны touching_side
        :param touching_side: сторона треугольника!, которой касается вневписанная окружность
        :return: радиус вневписанной окружности
        """
        return (self.AB_vector.length * self.AC_vector.length * sin(self.alpha_angle.rad) * 0.5) / (
                (self.AB_vector.length + self.BC_vector.length + self.AC_vector.length) * 0.5 - touching_side.length)

    def get_enscribed_center(self, touching_side: VectorWhistle, **kwargs):
        """
        Метод, возвращающий центр вневписанной окружности, касающийся стороны touching_side
        :param touching_side: сторона треугольника!, которой касается вневписанная окружность
        :return: центр вневписанной окружности
        """
        if touching_side in [self.AC_vector, self.CA_vector]:
            adjacent_gamma_bis = self.gamma_angle.get_adjacent(ground_vector=self.CB_vector).get_unit_bisector()
            adjacent_alpha_bis = self.alpha_angle.get_adjacent(ground_vector=self.AB_vector).get_unit_bisector()
            return adjacent_alpha_bis.get_intersection_with(adjacent_gamma_bis, **kwargs)
        elif touching_side in [self.BC_vector, self.CB_vector]:
            adjacent_beta_bis = self.beta_angle.get_adjacent(ground_vector=self.BA_vector).get_unit_bisector()
            adjacent_gamma_bis = self.gamma_angle.get_adjacent(ground_vector=self.CA_vector).get_unit_bisector()
            return adjacent_beta_bis.get_intersection_with(adjacent_gamma_bis, **kwargs)
        elif touching_side in [self.AB_vector, self.BA_vector]:
            adjacent_alpha_bis = self.alpha_angle.get_adjacent(ground_vector=self.AC_vector).get_unit_bisector()
            adjacent_beta_bis = self.beta_angle.get_adjacent(ground_vector=self.BC_vector).get_unit_bisector()
            return adjacent_alpha_bis.get_intersection_with(adjacent_beta_bis, **kwargs)
        else:
            raise ValueError
