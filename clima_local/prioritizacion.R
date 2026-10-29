# Conjunto minimo con prioritizr (Hanson et al. 2025) y HiGHS. Entradas de prioritizacion.py preparar.
# Escenarios: A especies en vacios; B especies; C especies + clima 17 %; D10/D17/D30 solo clima.
# Metas imposibles se recortan a lo disponible (red real + unidades candidatas).
suppressPackageStartupMessages({library(prioritizr)})
dst <- file.path(dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))),
                 "..", "resultados_v7", "09_prioritizacion")
ent <- file.path(dst, "entrada")

resolver <- function(esc, via, especies, clima, presupuesto = NA, meta_fija = NA, sufijo = "") {
  pu <- read.csv(file.path(ent, paste0("pu_", via, ".csv")))
  f <- read.csv(file.path(ent, paste0("features_", via, ".csv")))
  rij <- read.csv(file.path(ent, paste0("rij_", via, ".csv")))
  meta <- ifelse(f$tipo == "especie", if (!especies) 0 else if (is.na(meta_fija)) f$meta_especie * f$total_km2 else meta_fija * f$total_km2,
                 if (is.na(clima)) 0 else clima * f$total_km2)
  meta <- pmin(meta, 0.999 * f$disponible_km2)   # margen: igualar lo disponible falla por redondeo
  usar <- meta > 0
  f <- f[usar, ]; meta <- meta[usar]
  rij <- rij[rij$species %in% f$id, ]
  pu$id <- pu$id + 1; rij$pu <- rij$pu + 1   # prioritizr no admite id 0 (la red real); se devuelve al escribir
  pu$locked_in <- pu$id == 1
  t0 <- Sys.time()
  p <- problem(pu, f[, c("id", "name")], rij, cost_column = "cost") |>
    (\(q) if (is.na(presupuesto)) add_min_set_objective(q) else add_min_shortfall_objective(q, budget = presupuesto))() |>
    add_absolute_targets(meta) |>
    add_locked_in_constraints("locked_in") |>
    add_binary_decisions() |>
    add_highs_solver(gap = 0.005, time_limit = 3600, verbose = TRUE,
                     control = list(log_file = file.path(dst, paste0("log_highs_", esc, sufijo, ".txt"))))
  s <- solve(p, force = TRUE)
  seg <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  s$id <- s$id - 1
  write.csv(s[, c("id", "hybas_id", "cost", "solution_1")], file.path(dst, paste0("solucion_", esc, sufijo, ".csv")), row.names = FALSE)
  sel <- s$solution_1 == 1 & s$id > 0
  cat(sprintf("%s: %d features, %d unidades elegidas, %.0f km2, %.0f s\n", esc, nrow(f), sum(sel), sum(s$cost[sel]), seg))
  # Estado del solver por problema (status, brecha, objetivo, tiempo de HiGHS y total) para saber si la solucion es optima
  reg <- data.frame(escenario = paste0(esc, sufijo), fecha = format(Sys.time()), status = attr(s, "status"), gap = attr(s, "gap"),
                    objective = attr(s, "objective"), runtime_highs_s = attr(s, "runtime"), total_s = seg,
                    unidades = sum(sel), km2 = sum(s$cost[sel]))
  out <- file.path(dst, "estado_solver.csv")
  write.table(reg, out, sep = ",", row.names = FALSE, col.names = !file.exists(out), append = file.exists(out))
}

args <- commandArgs(TRUE)
suf <- paste(sub("^--sufijo=", "", grep("^--sufijo=", args, value = TRUE)), collapse = "")   # --sufijo=_recheck no sobrescribe soluciones
args <- grep("^--sufijo=", args, value = TRUE, invert = TRUE)
esc <- list(A = list("vacios", TRUE, NA), B = list("todo", TRUE, NA), C17 = list("todo", TRUE, 0.17),
            D10 = list("todo", FALSE, 0.10), D17 = list("todo", FALSE, 0.17), D30 = list("todo", FALSE, 0.30))
# Con presupuesto (min shortfall): 19 000 km2 (expansion modesta, 1.5 % del dominio) y la superficie de la curva voraz al 90 % (41 347 km2)
for (b in c(19000, 41347)) for (k in c("A", "B", "C17", "D17"))
  esc[[paste0(k, "_", round(b / 1000), "k")]] <- c(esc[[k]], b)
# Solo especies con una meta fija del 17 % o del 30 % de cada rango, en lugar de la meta de Rodrigues (Tabla S12)
for (b in c(19000, 41347)) for (m in c(17, 30))
  esc[[paste0("B", m, "f_", round(b / 1000), "k")]] <- list("todo", TRUE, NA, b, m / 100)
for (e in if (length(args)) args else names(esc)) do.call(resolver, c(list(e), esc[[e]], sufijo = suf))
