from manim import *
from numpy import *


class VerticeWhistle(Dot):
    def __init__(self, coordinates: ndarray, **kwargs):
        super().__init__(**kwargs)
        self.move_to(coordinates)

    def get_height_to(self, vector: 'VectorWhistle'):
        unit_normal = vector.get_unit_normal(start_vertice=self)
        H_vertice = unit_normal.get_intersection_with(vector)
        return VectorWhistle(self, H_vertice)

    def get_median_to(self, vector: 'VectorWhistle'):
        x = (vector.vertice_1.x + vector.vertice_2.x) / 2
        y = (vector.vertice_1.y + vector.vertice_2.y) / 2
        return VectorWhistle(self, VerticeWhistle(np.array([x, y, 0])))

    @property
    def coordinates(self):
        return self.get_center()

    @property
    def x(self):
        return self.get_center()[0]

    @property
    def y(self):
        return self.get_center()[1]

    def get_always_redraw(self) -> 'VectorWhistle':
        return always_redraw(lambda: VerticeWhistle(self.coordinates))


class VectorWhistle(Line):
    def __init__(self, vertice_1: VerticeWhistle, vertice_2: VerticeWhistle, **kwargs):
        super().__init__(vertice_1.coordinates, vertice_2.coordinates, **kwargs)

        self.vertice_1 = vertice_1
        self.vertice_2 = vertice_2

    def scalar_product_with(self, vector: 'VectorWhistle'):
        return self.coordinates[0] * vector.coordinates[0] + self.coordinates[1] * vector.coordinates[1]

    def get_reversed(self):
        return VectorWhistle(self.vertice_2, self.vertice_1)

    def get_intersection_with(self, vector: 'VectorWhistle'):
        # A_bisector = self.get_bisector_to_side(vertice=self.A_vertice)  # AA1
        # C_bisector = self.get_bisector_to_side(vertice=self.C_vertice)  # CC1
        # x = ((self.C_vertice.y - self.A_vertice.y) * A_bisector.x * C_bisector.x + A_bisector.y * C_bisector.x * self.A_vertice.x - A_bisector.x * C_bisector.y * self.C_vertice.x) / (A_bisector.y * C_bisector.x - A_bisector.x * C_bisector.y)
        # y = (x - self.A_vertice.x) / A_bisector.x * A_bisector.y + self.A_vertice.y
        # AA1 - self, CC1 - vector
        x = ((
                         vector.vertice_1.y - self.vertice_1.y) * self.x * vector.x + self.y * vector.x * self.vertice_1.x - self.x * vector.y * vector.vertice_1.x) / (
                        self.y * vector.x - self.x * vector.y)
        y = (x - self.vertice_1.x) / self.x * self.y + self.vertice_1.y
        return VerticeWhistle(coordinates=np.array([x, y, 0]))

    def get_line_center(self):
        return VerticeWhistle(
            np.array([(self.vertice_1.x + self.vertice_2.x) / 2, (self.vertice_1.y + self.vertice_2.y) / 2, 0]))

    def get_unit_normal(self, start_vertice: VerticeWhistle):
        # another_pos = VerticeWhistle(np.array([start_vertice.x + 1, start_vertice.y - self.coordinates[0] / self.coordinates[1], 0]))
        return VectorWhistle(start_vertice, VerticeWhistle(
            np.array([start_vertice.x + 1, start_vertice.y - self.coordinates[0] / self.coordinates[1], 0])))

    def get_always_redraw(self) -> 'VectorWhistle':
        return always_redraw(lambda: VectorWhistle(self.vertice_1, self.vertice_2))

    @property
    def coordinates(self):
        return np.array([self.vertice_2.x - self.vertice_1.x, self.vertice_2.y - self.vertice_1.y, 0])

    @property
    def x(self):
        return self.coordinates[0]

    @property
    def y(self):
        return self.coordinates[1]

    @property
    def length(self):
        return sqrt(self.coordinates[0] ** 2 + self.coordinates[1] ** 2)


class AngleWhistle(Angle):
    def __init__(self, vector_1: VectorWhistle, vector_2: VectorWhistle, **kwargs):
        super().__init__(vector_1, vector_2, **kwargs)

        self.vector_1 = vector_1
        self.vector_2 = vector_2

    @property
    def rad(self):
        return arccos(self.vector_1.scalar_product_with(self.vector_2) / (self.vector_1.length * self.vector_2.length))

    def get_unit_bisector(self):
        alpha = (self.vector_1.length * self.vector_2.x - self.vector_2.length * self.vector_1.x) / (
                    self.vector_2.length * self.vector_1.y - self.vector_1.length * self.vector_2.y)
        return VectorWhistle(self.vector_1.vertice_1, VerticeWhistle(
            np.array([self.vector_1.vertice_1.x + 1, self.vector_1.vertice_1.y + alpha, 0])))

    def get_bisector_to(self, vector: VectorWhistle):
        # ab - 1, ac - 2
        return VectorWhistle(self.vector_1.vertice_1, vector.get_intersection_with(self.get_unit_bisector()))

    def get_adjacent(self, ground_vector: VectorWhistle):
        if ground_vector == self.vector_1:
            new_vec = VectorWhistle(self.vector_1.vertice_1, VerticeWhistle(coordinates=np.array([self.vector_1.vertice_1.x + self.vector_1.get_reversed().x, self.vector_1.vertice_1.y + self.vector_1.get_reversed().y, 0])))
            return AngleWhistle(new_vec, self.vector_2)
        elif ground_vector == self.vector_2:
            new_vec = VectorWhistle(self.vector_2.vertice_1, VerticeWhistle(coordinates=np.array(
                [self.vector_2.vertice_1.x + self.vector_2.get_reversed().x,
                 self.vector_2.vertice_1.y + self.vector_2.get_reversed().y, 0])))
            return AngleWhistle(new_vec, self.vector_1)

    def get_always_redraw(self) -> 'AngleWhistle':
        return always_redraw(lambda: AngleWhistle(self.vector_1, self.vector_2))


class TriangleWhistle(Polygon):
    def __init__(self, A_vertice: VerticeWhistle, B_vertice: VerticeWhistle, C_vertice: VerticeWhistle, **kwargs):
        super().__init__(A_vertice.coordinates, B_vertice.coordinates, C_vertice.coordinates, **kwargs)

        self.A_vertice = A_vertice
        self.B_vertice = B_vertice
        self.C_vertice = C_vertice

        #self.AB_vector = always_redraw(lambda: VectorWhistle(self.A_vertice, self.B_vertice))
        #self.BC_vector = always_redraw(lambda: VectorWhistle(self.B_vertice, self.C_vertice))
        #self.CA_vector = always_redraw(lambda: VectorWhistle(self.C_vertice, self.A_vertice))
        #self.AC_vector = always_redraw(lambda: self.CA_vector.get_reversed())
        #self.BA_vector = always_redraw(lambda: self.AB_vector.get_reversed())
        #self.CB_vector = always_redraw(lambda: self.BC_vector.get_reversed())

        #self.alpha_angle = always_redraw(lambda: AngleWhistle(self.AB_vector, self.AC_vector))
        #self.beta_angle = always_redraw(lambda: AngleWhistle(self.BC_vector, self.BA_vector))
        #self.gamma_angle = always_redraw(lambda: AngleWhistle(self.CA_vector, self.CB_vector))


        self.AB_vector = VectorWhistle(self.A_vertice, self.B_vertice).get_always_redraw()
        self.BC_vector = VectorWhistle(self.B_vertice, self.C_vertice).get_always_redraw()
        self.CA_vector = VectorWhistle(self.C_vertice, self.A_vertice).get_always_redraw()
        self.AC_vector = self.CA_vector.get_reversed().get_always_redraw()
        self.BA_vector = self.AB_vector.get_reversed().get_always_redraw()
        self.CB_vector = self.BC_vector.get_reversed().get_always_redraw()

        self.alpha_angle = AngleWhistle(self.AB_vector, self.AC_vector)
        self.beta_angle = AngleWhistle(self.BC_vector, self.BA_vector)
        self.gamma_angle = AngleWhistle(self.CA_vector, self.CB_vector)

    def get_median_to_side(self, vertice: VerticeWhistle, **kwargs):
        if vertice.x == self.A_vertice.x:
            # BC
            return vertice.get_median_to(self.BC_vector)
        elif vertice.x == self.B_vertice.x:
            # AC
            return vertice.get_median_to(self.CA_vector)
        elif vertice.x == self.C_vertice.x:
            # AB
            return vertice.get_median_to(self.AB_vector)
        else:
            # nothing
            raise ValueError

    def get_height_to_side(self, vertice: VerticeWhistle, **kwargs):
        if vertice.x == self.A_vertice.x:
            # BC
            return vertice.get_height_to(self.BC_vector)
            # x = (self.BC_vector.x * self.BC_vector.y * (self.A_vertice.y - self.B_vertice.y) + self.BC_vector.x**2 * self.A_vertice.x + self.BC_vector.y**2 * self.B_vertice.x) / (self.BC_vector.x**2 + self.BC_vector.y**2)
            # y = (self.BC_vector.y * (x - self.B_vertice.x) / self.BC_vector.x + self.B_vertice.y)
        elif vertice.x == self.B_vertice.x:
            # AC
            return vertice.get_height_to(self.CA_vector.get_reversed())
            # x = (self.CA_vector.get_reversed().y * self.CA_vector.get_reversed().x * (self.B_vertice.y - self.A_vertice.y) + self.CA_vector.get_reversed().x ** 2 * self.B_vertice.x + self.CA_vector.get_reversed().y ** 2 * self.A_vertice.x) / (self.CA_vector.get_reversed().x ** 2 + self.CA_vector.get_reversed().y ** 2)
            # y = (x - self.A_vertice.x) / self.CA_vector.get_reversed().x * self.CA_vector.get_reversed().y + self.A_vertice.y
        elif vertice.x == self.C_vertice.x:
            # AB
            return vertice.get_height_to(self.AB_vector)
            # x = (self.AB_vector.y * self.AB_vector.x * (self.C_vertice.y - self.A_vertice.y) + self.AB_vector.x ** 2 * self.C_vertice.x + self.AB_vector.y ** 2 * self.A_vertice.x) / (self.AB_vector.x ** 2 + self.AB_vector.y ** 2)
            # y = (x - self.A_vertice.x) / self.AB_vector.x * self.AB_vector.y + self.A_vertice.y
        else:
            # nothing
            raise ValueError
        # AC
        # x = (self.CA_vector.get_reversed().y * self.CA_vector.get_reversed().x * (self.B_vertice.y - self.A_vertice.y) + self.CA_vector.get_reversed().x**2 * self.B_vertice.x + self.CA_vector.get_reversed().y**2 * self.A_vertice.x) / (self.CA_vector.get_reversed().x**2 + self.CA_vector.get_reversed().y**2)
        # y = (x - self.A_vertice.x) / self.CA_vector.get_reversed().x * self.CA_vector.get_reversed().y + self.A_vertice.y

        # BC
        # x = (self.BC_vector.x * self.BC_vector.y * (self.A_vertice.y - self.B_vertice.y) + self.BC_vector.x**2 * self.A_vertice.x + self.BC_vector.y**2 * self.B_vertice.x) / (self.BC_vector.x**2 + self.BC_vector.y**2)
        # y = (self.BC_vector.y * (x - self.B_vertice.x) / self.BC_vector.x + self.B_vertice.y)

        # AB
        # x = (self.AB_vector.y * self.AB_vector.x * (self.C_vertice.y - self.A_vertice.y) + self.AB_vector.x**2 * self.C_vertice.x + self.AB_vector.y**2 * self.A_vertice.x) / (self.AB_vector.x**2 + self.AB_vector.y**2)
        # y = (x - self.A_vertice.x) / self.AB_vector.x * self.AB_vector.y + self.A_vertice.y
        # return VectorWhistle(vertice, VerticeWhistle(coordinates=np.array([x, y, 0])), **kwargs)

    def get_bisector_to_side(self, vertice: VerticeWhistle, **kwargs):
        if vertice.x == self.A_vertice.x:
            # BC
            return self.alpha_angle.get_bisector_to(self.BC_vector)
            # dependence = (self.AB_vector.length * self.CA_vector.get_reversed().x - self.CA_vector.get_reversed().length * self.AB_vector.x) / (self.CA_vector.get_reversed().length * self.AB_vector.y - self.AB_vector.length * self.CA_vector.get_reversed().y)
            # x = (self.BC_vector.x * (self.B_vertice.y - self.A_vertice.y + dependence * self.A_vertice.x) - self.BC_vector.y * self.B_vertice.x) / (self.BC_vector.x * dependence - self.BC_vector.y)
            # y = dependence * (x - self.A_vertice.x) + self.A_vertice.y
        elif vertice.x == self.B_vertice.x:
            # AC
            return self.beta_angle.get_bisector_to(self.CA_vector.get_reversed())
            # dependence = (self.BC_vector.length * self.AB_vector.get_reversed().x - self.AB_vector.get_reversed().length * self.BC_vector.x) / (self.AB_vector.get_reversed().length * self.BC_vector.y - self.BC_vector.length * self.AB_vector.get_reversed().y)
            # x = (self.CA_vector.get_reversed().x * (self.A_vertice.y - self.B_vertice.y + dependence * self.B_vertice.x) - self.CA_vector.get_reversed().y * self.A_vertice.x) / (dependence * self.CA_vector.get_reversed().x - self.CA_vector.get_reversed().y)
            # y = self.B_vertice.y + dependence * (x - self.B_vertice.x)
        elif vertice.x == self.C_vertice.x:
            # AB
            return self.gamma_angle.get_bisector_to(self.AB_vector)
            # dependence = (self.BC_vector.get_reversed().length * self.CA_vector.x - self.CA_vector.length * self.BC_vector.get_reversed().x) / (self.CA_vector.length * self.BC_vector.get_reversed().y - self.BC_vector.get_reversed().length * self.CA_vector.y)
            # x = (self.AB_vector.x * (self.A_vertice.y - self.C_vertice.y + self.C_vertice.x * dependence) - self.AB_vector.y * self.A_vertice.x) / (dependence * self.AB_vector.x - self.AB_vector.y)
            # y = (x - self.C_vertice.x) * dependence + self.C_vertice.y
        else:
            # nothing
            raise ValueError
        # BC
        # dependence = (self.AB_vector.length * self.CA_vector.get_reversed().x - self.CA_vector.get_reversed().length * self.AB_vector.x) / (self.CA_vector.get_reversed().length * self.AB_vector.y - self.AB_vector.length * self.CA_vector.get_reversed().y)
        # x = (self.BC_vector.x * (self.B_vertice.y - self.A_vertice.y + dependence * self.A_vertice.x) - self.BC_vector.y * self.B_vertice.x) / (self.BC_vector.x * dependence - self.BC_vector.y)
        # y = dependence * (x - self.A_vertice.x) + self.A_vertice.y
        # return VectorWhistle(self.A_vertice, VerticeWhistle(coordinates=np.array([x, y, 0])), **kwargs)

        # AC
        # dependence = (self.BC_vector.length * self.AB_vector.get_reversed().x - self.AB_vector.get_reversed().length * self.BC_vector.x) / (self.AB_vector.get_reversed().length * self.BC_vector.y - self.BC_vector.length * self.AB_vector.get_reversed().y)
        # x = (self.CA_vector.get_reversed().x * (self.A_vertice.y - self.B_vertice.y + dependence * self.B_vertice.x) - self.CA_vector.get_reversed().y * self.A_vertice.x) / (dependence * self.CA_vector.get_reversed().x - self.CA_vector.get_reversed().y)
        # y = self.B_vertice.y + dependence * (x - self.B_vertice.x)
        # return VectorWhistle(self.B_vertice, VerticeWhistle(coordinates=np.array([x, y, 0])), **kwargs)

        # AB
        # dependence = (self.BC_vector.get_reversed().length * self.CA_vector.x - self.CA_vector.length * self.BC_vector.get_reversed().x) / (self.CA_vector.length * self.BC_vector.get_reversed().y - self.BC_vector.get_reversed().length * self.CA_vector.y)
        # x = (self.AB_vector.x * (self.A_vertice.y - self.C_vertice.y + self.C_vertice.x * dependence) - self.AB_vector.y * self.A_vertice.x) / (dependence * self.AB_vector.x - self.AB_vector.y)
        # y = (x - self.C_vertice.x) * dependence + self.C_vertice.y
        # return VectorWhistle(vertice, VerticeWhistle(coordinates=np.array([x, y, 0])), **kwargs)

    def get_incenter(self, **kwargs):
        # A_bisector = self.get_bisector_to_side(vertice=self.A_vertice)  # AA1
        # C_bisector = self.get_bisector_to_side(vertice=self.C_vertice)  # CC1
        # x = ((self.C_vertice.y - self.A_vertice.y) * A_bisector.x * C_bisector.x + A_bisector.y * C_bisector.x * self.A_vertice.x - A_bisector.x * C_bisector.y * self.C_vertice.x) / (A_bisector.y * C_bisector.x - A_bisector.x * C_bisector.y)
        # y = (x - self.A_vertice.x) / A_bisector.x * A_bisector.y + self.A_vertice.y
        # return VerticeWhistle(coordinates=np.array([x, y, 0]), **kwargs)
        return self.get_bisector_to_side(vertice=self.A_vertice).get_intersection_with(
            self.get_bisector_to_side(vertice=self.B_vertice))

    def get_circumscribed_center(self):
        return self.AB_vector.get_unit_normal(start_vertice=self.AB_vector.get_line_center()).get_intersection_with(
            self.BC_vector.get_unit_normal(start_vertice=self.BC_vector.get_line_center()))

    def get_incenter_radius(self):
        return (self.AB_vector.length * self.CA_vector.length * sin(self.alpha_angle.rad)) / (
                    self.AB_vector.length + self.BC_vector.length + self.CA_vector.length)

    def get_circumscribed_radius(self):
        return self.BC_vector.length / (2 * sin(self.alpha_angle.rad))

    def get_enscribed_radius(self, touching_side: VectorWhistle):
        return (self.AB_vector.length * self.AC_vector.length * sin(self.alpha_angle.rad) * 0.5) / (
                (self.AB_vector.length + self.BC_vector.length + self.AC_vector.length) * 0.5 - touching_side.length)

    def get_enscribed_center(self, touching_side: VectorWhistle):
        if touching_side in [self.AC_vector, self.CA_vector]:
            adjacent_gamma_bis = self.gamma_angle.get_adjacent(ground_vector=self.CB_vector).get_unit_bisector()
            adjacent_alpha_bis = self.alpha_angle.get_adjacent(ground_vector=self.AB_vector).get_unit_bisector()
            return adjacent_alpha_bis.get_intersection_with(adjacent_gamma_bis)
        elif touching_side in [self.BC_vector, self.CB_vector]:
            adjacent_beta_bis = self.beta_angle.get_adjacent(ground_vector=self.BA_vector).get_unit_bisector()
            adjacent_gamma_bis = self.gamma_angle.get_adjacent(ground_vector=self.CA_vector).get_unit_bisector()
            return adjacent_beta_bis.get_intersection_with(adjacent_gamma_bis)
        elif touching_side in [self.AB_vector, self.BA_vector]:
            adjacent_alpha_bis = self.alpha_angle.get_adjacent(ground_vector=self.AC_vector).get_unit_bisector()
            adjacent_beta_bis = self.beta_angle.get_adjacent(ground_vector=self.BC_vector).get_unit_bisector()
            return adjacent_alpha_bis.get_intersection_with(adjacent_beta_bis)
        else:
            raise ValueError


class TestMyOwnClasses(MovingCameraScene):
    def construct(self):
        number_plane = NumberPlane(x_range=[-50, 50, 1], y_range=[-50, 50, 1],
                                   background_line_style={"stroke_color": GREY, "stroke_width": 1,
                                                          "stroke_opacity": 0.6},
                                   axis_config={"color": GREY, "stroke_opacity": 0.6})
        self.play(Create(number_plane))
        self.wait()
        A = VerticeWhistle(coordinates=np.array([-3, 2, 0]))
        B = VerticeWhistle(coordinates=np.array([0, -2, 0]))
        C = VerticeWhistle(coordinates=np.array([3, 3, 0]))
        triangle = TriangleWhistle(A, B, C)
        incenter = always_redraw(lambda: triangle.get_incenter())
        incenter_circle = always_redraw(lambda: Circle(triangle.get_incenter_radius()).move_to(incenter))

        vec_1 = always_redraw(lambda: triangle.gamma_angle.get_adjacent(ground_vector=triangle.CB_vector).vector_1)
        vec_2 = always_redraw(lambda: triangle.alpha_angle.get_adjacent(ground_vector=triangle.AB_vector).vector_1)
        AC_center = always_redraw(lambda: triangle.get_enscribed_center(touching_side=triangle.AC_vector))
        AC_circle = always_redraw(lambda: Circle(triangle.get_enscribed_radius(triangle.AC_vector)).move_to(AC_center))

        vec_3 = always_redraw(lambda: triangle.beta_angle.get_adjacent(ground_vector=triangle.BA_vector).vector_1)
        vec_4 = always_redraw(lambda: triangle.gamma_angle.get_adjacent(ground_vector=triangle.CA_vector).vector_2)
        BC_center = always_redraw(lambda: triangle.get_enscribed_center(triangle.BC_vector))
        BC_circle = always_redraw(lambda: Circle(triangle.get_enscribed_radius(triangle.BC_vector)).move_to(BC_center))

        vec_5 = always_redraw(lambda: triangle.alpha_angle.get_adjacent(ground_vector=triangle.AC_vector).vector_1)
        vec_6 = always_redraw(lambda: triangle.beta_angle.get_adjacent(ground_vector=triangle.BC_vector).vector_1)
        AB_center = always_redraw(lambda: triangle.get_enscribed_center(triangle.AB_vector))
        AB_circle = always_redraw(lambda: Circle(triangle.get_enscribed_radius(triangle.AB_vector)).move_to(AB_center))

        self.play(Create(triangle.A_vertice), Create(triangle.B_vertice), Create(triangle.C_vertice))
        self.wait()
        self.play(Create(triangle.AB_vector), Create(triangle.BC_vector), Create(triangle.AC_vector))
        self.wait()
        self.play(self.camera.frame.animate.scale(4))
        self.wait()
        self.play(Create(vec_1), Create(vec_2), Create(vec_3), Create(vec_4), Create(vec_5), Create(vec_6))
        self.wait()
        self.play(Create(incenter), Create(AC_center), Create(BC_center), Create(AB_center))
        self.wait()
        self.play(Create(incenter_circle), Create(AC_circle), Create(BC_circle), Create(AB_circle))
        self.wait()
        self.play(self.camera.frame.animate.scale(1.25))
        self.wait()
        self.play(ApplyMethod(A.shift, 3*LEFT, run_time=3))
        self.play(ApplyMethod(B.shift, 3*DOWN, run_time=3))
        self.play(ApplyMethod(C.shift, 3*RIGHT, run_time=3))



