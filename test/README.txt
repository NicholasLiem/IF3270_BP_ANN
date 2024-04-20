Seperti sebelumnya ada dua bagian yaitu case dan expect.

Isi dari case:
- model.input_size.
- model.layers, terdiri dari daftar objek beratribut berikut:
    - number_of_neurons.
    - activation_function (nilai valid: linear, relu, sigmoid, softmax).
- input:
    - Dimensi 1 menandakan vektor ke-i. Ukurannya arbitrary
    - Dimensi 2 menandakan isi suatu vektor. Ukurannya sesuai dengan model.input_size.
- weights:
    - Dimensi 1 bersesuaian dengan layer di layers.
    - Dimensi 2 berukuran banyak neuron pada lapisan sebelumnya + 1.
        - Khusus untuk layer pertama: model.input_size + 1.
        - Baris pertama adalah bias.
    - Dimensi 3 berukuran banyak neuron pada lapisan yang bersesuaian.
- target:
    - Ukuran dimensi 1 sesuai dengan ukuran dimensi 1 dari input.
    - Ukuran dimensi 2 sesuai dengan banyak neuron pada lapisan terakhir.
- learning_parameters:
    - learning_rate
    - batch_size
    - max_iteration
    - error_threshold: Perhitungan error sesuai dengan fungsi aktivasi pada lapisan terakhir. Fungsi log jika mengacu pada spesifikasi berbasis e (ekuivalen dengan ln(x)).

Isi dari expect:
- stopped_by
     Hanya ada dua nilai yang valid: max_iteration dan error_threshold.
     Jika bernilai max_iteration, diharapkan pembelajaran berhenti karena banyak iterasinya mencapai maksimum.
     Jika bernilai error_threshold, diharapkan pembelajaran berhenti karena rata-rata nilai eror yang diperoleh pada iterasi terakhir lebih kecil atau sama dengan error threshold.
- final_weights:
     Jika atribut ini dinyatakan, kesamaan weights dicek setelah semua iterasi pembelajaran dilakukan.
     Spesifikasi dimensi sama dengan case.initial_weights.
     Maksimum SSE (sum of squares error) yang dapat diterima adalah 10^-7.