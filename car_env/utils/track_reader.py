import numpy as np
import pathlib
import scipy
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
# fmt: on


class TrackReader:

    def __init__(self, filename, flip=False, reverse=False):
        track_path = os.path.join(os.path.dirname(__file__),
                                  "../envs/simulators/tracks/" + filename + ".csv")
        self.file = pathlib.Path(track_path)
        data = np.loadtxt(self.file, delimiter=",")
        if not reverse:
            data = np.flip(data, axis=0)

        # if negative values are present, shift the track to positive values
        # if np.any(data[:, 0] < 0):
        #     data[:, 0] += np.abs(np.min(data[:, 0]))
        if np.any(data[:, 1] < 0):
            data[:, 1] += np.abs(np.min(data[:, 1]))

        # spline smoothing factor in scipy.interpolate.splprep
        self.splprep_s = 0.3

        self.x = data[:, 0]
        if not flip:
            self.y = data[:, 1]
        else:
            self.y = -data[:, 1]
        self.w_r = data[:, 2]
        self.w_l = data[:, 3]

    def preprocess_track(self, debug=False, plot=False):

        x, y, w_r, w_l = self.x, self.y, self.w_r, self.w_l

        # smooth x and y with savgol filter
        w_r = scipy.signal.savgol_filter(w_r, 10, 3)

        track_width = w_r + w_l

        w_splprep = 1 / track_width

        tck, u = scipy.interpolate.splprep(
            [x, y], w_splprep, s=self.splprep_s, k=5, per=len(x)
        )
        self.tck = tck

        x_s, y_s = scipy.interpolate.splev(u, tck)

        e = np.sqrt((x_s - x) ** 2 + (y_s - y) ** 2)
        track_width_corrected = (track_width / 2 - e) * 2

        rmse = np.sqrt(np.mean((x_s - x) ** 2 + (y_s - y) ** 2))
        self.rmse = rmse

        x_dot, y_dot = scipy.interpolate.splev(u, tck, der=1)
        self.x_dot = x_dot
        self.y_dot = y_dot
        self.u = u

        x_ddot, y_ddot = scipy.interpolate.splev(u, tck, der=2)
        self.x_ddot = x_ddot
        self.y_ddot = y_ddot

        curvature = (x_dot * y_ddot - y_dot * x_ddot) / (x_dot**2 + y_dot**2) ** (3 / 2)

        self.curvature = curvature

        path_s = scipy.integrate.cumulative_trapezoid(np.sqrt(x_dot**2 + y_dot**2), u, initial=0)
        self.s = path_s

        # x_s[:] -= x_s[0]
        # y_s[:] -= y_s[0]

        # interpolate all values to 10 points per meter
        N = int(path_s[-1] * 10)
        path_s = np.linspace(0, path_s[-1], N)
        x, y = scipy.interpolate.splev(path_s / path_s[-1], tck)
        x_dot, y_dot = scipy.interpolate.splev(path_s / path_s[-1], tck, der=1)
        x_ddot, y_ddot = scipy.interpolate.splev(path_s / path_s[-1], tck, der=2)
        curvature = (x_dot * y_ddot - y_dot * x_ddot) / (x_dot**2 + y_dot**2) ** (3 / 2)
        track_width = np.interp(path_s, self.s, track_width)
        track_width_corrected = np.interp(path_s, self.s, track_width_corrected)

        # calucate heading angle for each point
        heading = np.arctan2(y_dot, x_dot)

        # acumulate curvature to get heading angle
        heading_curv = scipy.integrate.cumulative_trapezoid(curvature, path_s, initial=0)

        if plot:
            heading_unwind = np.unwrap(heading)
            plt.figure()
            plt.subplot(5, 1, 1)
            plt.plot(x, y, "r", label="track points")
            # plot arrow for track direction
            plt.arrow(
                x[0],
                y[0],
                x[1],
                y[1],
                head_width=2,
                alpha=0.20,
                color="orange",
            )

            plt.subplot(5, 1, 2)
            plt.plot(path_s, heading_unwind - heading_unwind[0], label="heading")  
            plt.subplot(5, 1, 3)
            plt.plot(path_s, curvature, label="curvature")
            plt.subplot(5, 1, 4)
            plt.plot(path_s, heading_curv, label="heading_curv width")
            plt.subplot(5, 1, 5)
            plt.plot(path_s, x, label="x")
            plt.tight_layout()
            plt.show()

        if debug:
            return (
                path_s,
                x,
                y,
                track_width_corrected,
                curvature,
                x_s,
                y_s,
                track_width,
                u,
            )
        else:
            return path_s, x, y, track_width, curvature, heading

    def plot_track(self, plot_inport_dots=False, ax=None, n=None):

        path_s, x, y, track_width_corrected, curvature, x_s, y_s, track_width, u = (
            self.preprocess_track(debug=True)
        )
        print(f"Track length: {path_s[-1]} m")
        print(f"{len(x)} points")
        plt.rcParams["figure.figsize"] = (16, 8)
        plt.figure()
        # start arrow
        plt.arrow(
            x_s[0],
            y_s[0],
            x_s[1] - x_s[0],
            y_s[1] - y_s[0],
            head_width=2,
            alpha=0.20,
            color="orange",
        )

        if plot_inport_dots:
            plt.scatter(x, y)
        # print type x_s and y_s
        print("x_s shape: ", x_s.shape, "y_s shape: ", y_s.shape)
        plt.scatter(x, y, color="r", label="track points")

        plt.plot(x_s, y_s, "r", label="centerline spline")
        plt.title(f"Track map {self.file.name}", fontsize=20)

        for i in range(len(x)):
            if plot_inport_dots:
                circle = plt.Circle(
                    (x[i], y[i]),
                    track_width[i] / 2,
                    color="g",
                    fill=False,
                    alpha=0.15,
                )
                plt.gcf().gca().add_artist(circle)

            circle_2 = plt.Circle(
                (x[i], y[i]),
                track_width_corrected[i] / 2,
                color="b",
                fill=False,
                alpha=0.15,
            )
            plt.gcf().gca().add_artist(circle_2)

        for i in range(0, len(x_s), 20):
            plt.text(x_s[i], y_s[i] + 0.2, f"{path_s[i]:.0f} ", fontsize=12)

        plt.grid()
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.tight_layout()
        plt.legend(["centerline spline", "track direction"])

    def to_Cartesian(self, s: np.ndarray, n: np.ndarray):
        u = s / self.s[-1]
        x, y = scipy.interpolate.splev(u, self.tck)
        x_dot, y_dot = scipy.interpolate.splev(u, self.tck, der=1)
        m = np.hypot(x_dot, y_dot)
        N = np.array([-y_dot, x_dot]) / m
        x_out = x + n * N[0]
        y_out = y + n * N[1]
        return x_out, y_out

    def plot_points(
        self, s: np.ndarray, n: np.ndarray, v_x=None, marker="o", alpha=1.0
    ):
        x_out, y_out = self.to_Cartesian(s, n)
        if v_x is not None:
            plt.scatter(x_out, y_out, c=v_x, cmap="jet", marker=marker, alpha=alpha)
            plt.colorbar().set_label("v_x [m/s]")
        else:
            plt.scatter(x_out, y_out, color="r", marker=marker, alpha=alpha)

    def plot_points_cartesian(self, x, y, v_x=None):
        if v_x is not None:
            plt.scatter(x, y, c=v_x, cmap="jet")
            plt.colorbar().set_label("v_x [m/s]")
        else:
            plt.scatter(x, y, color="r")

    def length(self):
        return self.s[-1]

    def file_path(self) -> pathlib.Path:
        return self.file

    def track_name(self) -> str:
        return self.file.stem

    def plot_curvature(self):
        plt.figure()
        plt.plot(self.s, self.curvature, label="curvature")
        plt.plot(self.s, self.w_l, label="width")
        plt.xlabel("s [m]")
        plt.ylabel("curvature [1/m]")
        plt.title("Curvature")
        plt.grid()
        plt.legend()
        plt.tight_layout()

    def plot_max_width(self):
        plt.figure()
        plt.plot(self.s, self.w_l, label="width")
        inv_k = 1 / np.abs(self.curvature)
        inv_k = np.clip(inv_k, 0, 2)
        plt.plot(self.s, inv_k, label="1/curvature")
        plt.xlabel("s [m]")
        plt.ylabel("width [m]")
        plt.title("Track width")
        plt.legend()
        plt.grid()
        plt.tight_layout()


if __name__ == "__main__":

    track = TrackReader("./envs/simulators/tracks/icra_2023.csv")
    track.plot_track(plot_inport_dots=False)
    # track.plot_curvature()
    # track.plot_max_width()
    s = np.linspace(0, track.length(), 1000)
    n = np.ones_like(s) * 0.05
    print(f"RMSE {track.rmse}")
    # track.plot_points(s, n)
    print(track.track_name())
    p = pathlib.Path("results") / (track.track_name() + ".csv")
    print(p)

    plt.show()
