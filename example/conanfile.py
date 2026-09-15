from conan import ConanFile


class ExampleApp(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("food_volume_measure/0.1.0")
