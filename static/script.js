document.addEventListener('DOMContentLoaded', () => {
  const menuToggle = document.querySelector('.menu-toggle');
  const navLinks = document.querySelector('.nav-links');

  menuToggle.addEventListener('click', () => {
      navLinks.classList.toggle('nav-active');
      menuToggle.classList.toggle('hide-toggle'); // اضافه کردن یک کلاس به دکمه همبرگری
  });

  // اضافه کردن رویداد کلیک برای هر عنصر منو برای پنهان کردن منو پس از انتخاب یک گزینه
  navLinks.querySelectorAll('a').forEach(link => {
      link.addEventListener('click', () => {
          navLinks.classList.remove('nav-active');
          menuToggle.classList.add('hide-toggle');
      });
  });

  let current = 1,
      playPauseBool = true,
      interval;

  const changeSlides = () => {
      const slideList = document.querySelectorAll(".slide");
      const slides = Array.from(slideList);
      if (current > slides.length) {
          current = 1;
      } else if (current === 0) {
          current = slides.length;
      }
      slides.forEach(slide => {
          if (slide.classList[1].split("-")[1] * 1 === current) {
              slide.style.cssText = "visibility: visible; opacity: 1";
          } else {
              slide.style.cssText = "visibility: hidden; opacity: 0";
          }
      });
  };

  const arrowVisibility = () => {
      const arrows = document.querySelectorAll(".control");
      Array.from(arrows).forEach(arrow => {
          if (!playPauseBool) {
              arrow.classList.add("arrows-visibility");
          } else {
              arrow.classList.remove("arrows-visibility");
          }
      });
  };

  const changePlayPause = () => {
      const i = document.querySelector(".play-pause i");
      const cls = i.classList[1];
      if (cls === "fa-play") {
          i.classList.remove("fa-play");
          i.classList.add("fa-pause");
      } else {
          i.classList.remove("fa-pause");
          i.classList.add("fa-play");
      }
  };

  const playPause = () => {
      if (playPauseBool) {
          interval = setInterval(() => {
              current++;
              changeSlides();
          }, 3000);
          playPauseBool = false;
      } else {
          clearInterval(interval);
          playPauseBool = true;
      }
      arrowVisibility();
      changePlayPause();
  };

  document.querySelector(".left-arrow").addEventListener("click", () => {
      if (!playPauseBool) {
          playPause();
      }
      current--;
      changeSlides();
  });

  document.querySelector(".right-arrow").addEventListener("click", () => {
      if (!playPauseBool) {
          playPause();
      }
      current++;
      changeSlides();
  });

  document.querySelector(".play-pause").addEventListener("click", () => {
      playPause();
  });

  changeSlides();
  playPause();
});
document.addEventListener('DOMContentLoaded', function() {
  // گرفتن لیست همه منوهای زیری
  var subMenus = document.querySelectorAll('.sub-menu');

  // اضافه کردن رویداد کلیک به هر منوی زیری
  subMenus.forEach(function(subMenu) {
      // وقتی که بر روی هر منوی زیری کلیک می‌شود
      subMenu.parentElement.querySelector('a').addEventListener('click', function(event) {
          event.preventDefault(); // جلوگیری از اجرای عملیات پیش‌فرض کلیک بر روی لینک

          // تغییر وضعیت نمایش منوی زیری
          if (subMenu.style.display === 'block') {
              subMenu.style.display = 'none';
          } else {
              subMenu.style.display = 'block';
          }
      });
  });
});
