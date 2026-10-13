"""
NGSolve finite-element temperature solver.
"""

import os
from ngsolve import H1, BilinearForm, LinearForm, GridFunction, grad, dx, ds, TaskManager, VTKOutput, SetNumThreads


def solve_temperature(cfg, mesh):
    SetNumThreads(int(cfg.cpu_threads))
    fes = H1(mesh, order=cfg.order)
    u, v = fes.TnT()

    a = BilinearForm(fes, symmetric=True)
    a += cfg.kappa * grad(u) * grad(v) * dx
    a += cfg.h_hole * u * v * ds("hole")

    f = LinearForm(fes)
    f += cfg.q_top * v * ds("top")
    f += cfg.h_hole * cfg.T_inf * v * ds("hole")

    gfu = GridFunction(fes)

    with TaskManager():
        a.Assemble()
        f.Assemble()
        gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec

    return gfu


def export_vtk(cfg, mesh, gfu, name):
    vtk = VTKOutput(
        mesh,
        coefs=[gfu],
        names=["temperature"],
        filename=os.path.join(cfg.outdir, name),
        subdivision=2,
    )
    vtk.Do()
