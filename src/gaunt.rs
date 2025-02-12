pub(crate) fn beta_k(k: u64, m: i64) -> f64 {
    let k = k as f64;
    let m = m as f64;

    ((k * k - m * m) / (4. * k * k - 1.)).sqrt()
}

pub(crate) fn q_kl1(k: u64, l: u64, m: i64) -> f64 {
    (-1.0_f64).powi((m % 2) as i32)
        * if k == l + 1 {
            beta_k(k, m)
        } else if l == k + 1 {
            beta_k(l, m)
        } else {
            0.
        }
}

pub(crate) fn q_kl2(k: u64, l: u64, m: i64) -> f64 {
    (3. / 2.)
        * if k == l {
            beta_k(k + 1, m).powi(2) + beta_k(k, m).powi(2) - 1. / 3.
        } else if k == l + 2 {
            beta_k(k, m) * beta_k(l + 1, m)
        } else if k + 2 == l {
            beta_k(k + 1, m) * beta_k(l, m)
        } else {
            0.
        }
}

pub(crate) fn q_kl1_h(k: u64, l: u64, m: i64) -> f64 {
    let lambda_k = (k * (k + 1)) as f64;
    let lambda_l = (l * (l + 1)) as f64;

    q_kl1(k, l, m) * ((lambda_k + lambda_l) / 2. - 1.)
}

pub(crate) fn q_kl2_h(k: u64, l: u64, m: i64) -> f64 {
    let lambda_k = (k * (k + 1)) as f64;
    let lambda_l = (l * (l + 1)) as f64;

    q_kl2(k, l, m) * ((lambda_k + lambda_l) / 2. - 3.)
}

pub(crate) fn q_kl1_hd(k: u64, l: u64, m: i64) -> f64 {
    let lambda_k = (k * (k + 1)) as f64;
    let lambda_l = (l * (l + 1)) as f64;

    q_kl1(k, l, m) * ((lambda_l - lambda_k) / 2. + 1.)
}

pub(crate) fn q_kl2_hd(k: u64, l: u64, m: i64) -> f64 {
    let lambda_k = (k * (k + 1)) as f64;
    let lambda_l = (l * (l + 1)) as f64;

    q_kl2(k, l, m) * ((lambda_l - lambda_k) / 2. + 3.)
}
