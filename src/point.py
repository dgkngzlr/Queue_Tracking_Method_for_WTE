import math


class Dist:
    def dot(self, v, w):
        x, y, z = v
        X, Y, Z = w
        return x * X + y * Y + z * Z

    def length(self, v):
        x, y, z = v
        return math.sqrt(x * x + y * y + z * z)

    def vector(self, b, e):
        x, y, z = b
        X, Y, Z = e
        return (X - x, Y - y, Z - z)

    def unit(self, v):
        x, y, z = v
        mag = self.length(v)
        return (x / mag, y / mag, z / mag)

    def distance(self, p0, p1):
        return self.length(self.vector(p0, p1))

    def scale(self, v, sc):
        x, y, z = v
        return (x * sc, y * sc, z * sc)

    def add(self, v, w):
        x, y, z = v
        X, Y, Z = w
        return (x + X, y + Y, z + Z)

    def pnt2line(self, pnt, start, end):  # pnt,start,end params have 3 dimensions x,y,z Z must be 0 for img processing
        line_vec = self.vector(start, end)  # all the points are going to transform to the vector space
        pnt_vec = self.vector(start, pnt)
        line_len = self.length(line_vec)  # finds the length from the vector
        line_unitvec = self.unit(line_vec)  # finds the corresponding unit value of x,y,z coordinates
        pnt_vec_scaled = self.scale(pnt_vec,
                                    1.0 / line_len)  # scales the point vector to the line vector an returns scaled values in vector form
        t = self.dot(line_unitvec, pnt_vec_scaled)  # mulitplies the two vector values
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        nearest = self.scale(line_vec, t)
        dist = self.distance(nearest, pnt_vec)
        return dist

    def listcheck(self, queue, start, end):
        temp_queue = []
        for person in queue:
            dist = self.pnt2line((person.center_x, person.center_y, 0), (start[0], start[1], 0), (end[0], end[1], 0))
            if dist < 200:
                temp_queue.append(person)
            else:
                pass
                #print("Person at these coordinates is not eligible", person.center_x, person.center_y)
        return temp_queue
